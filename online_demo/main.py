import os
import io
import tvm
import time
import cv2
import onnx
import torch
import torch.onnx
import torchvision
import numpy as np
import tvm.relay as relay
from PIL import Image
import tvm.contrib.graph_runtime as graph_runtime
from mobilenet_v2_tsm import MobileNetV2

HISTORY_LOGIT = True


def torch2tvm_module(model, inputs, target):
    model.eval()
    input_names = []
    input_shapes = {}
    with torch.no_grad():
        for idx, input in enumerate(inputs):
            name = "i" + str(idx)
            input_names.append(name)
            input_shapes[name] = input.shape
        buffer = io.BytesIO()
        torch.onnx.export(model, inputs, buffer, input_names=input_names,
                          output_names=["o" + str(i) for i in range(len(inputs))])  # torch to onnx model
        buffer.seek(0, 0)
        onnx_model = onnx.load_model(buffer)  # load onnx model
        relay_module, params = relay.frontend.from_onnx(onnx_model, shape=input_shapes)  # params(weights)
    with relay.build_config(opt_level=3):
        graph, tvm_module, params = relay.build(relay_module, target, params=params)

    return graph, tvm_module, params


def torch2executor(model, inputs, target):
    prefix = f"mobilenetv2_tsm_tvm_{target}"
    lib_fname = f'{prefix}.tar'
    graph_fname = f'{prefix}.json'
    params_fname = f'{prefix}.params'
    if os.path.exists(lib_fname) and os.path.exists(graph_fname) and os.path.exists(params_fname):
        with open(graph_fname, 'rt') as f:
            graph = f.read()
        tvm_module = tvm.module.load(lib_fname)
        params = relay.load_param_dict(bytearray(open(params_fname, 'rb').read()))
    else:
        graph, tvm_module, params = torch2tvm_module(model, inputs, target)
        tvm_module.export_library(lib_fname)
        with open(graph_fname, 'wt') as f:
            f.write(graph)
        with open(params_fname, 'wb') as f:
            f.write(relay.save_param_dict(params))

    ctx = tvm.gpu() if target.startswith('cuda') else tvm.cpu()
    graph_module = graph_runtime.create(graph, tvm_module, ctx)  # graph json, tvm module and tvm context
    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs):
        for idx, value in enumerate(inputs):
            graph_module.set_input(idx, value)
        graph_module.run()
        return tuple(graph_module.get_output(idx) for idx in range(len(inputs)))

    return executor, ctx


def get_executor():
    model = MobileNetV2(n_class=27)
    mobilenetv2_jester = torch.load('mobilenetv2_jester.pth.tar')['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in mobilenetv2_jester.items():
        name = k[7:]
        if 'new_fc' in name:
            name = name.replace('new_fc', 'classifier')
        else:
            if 'net' in name:
                name = name.replace('net.', '')
            name = name[11:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    inputs = (torch.rand(1, 3, 224, 224),
              torch.zeros([1, 3, 56, 56]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 20, 7, 7]),
              torch.zeros([1, 20, 7, 7]))
    return torch2executor(model, inputs, target='cuda')


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
        else:
            return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # from HWC to CHW format
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return tensor


def transform(frame):
    # from H*W*C to 1*C*H*W
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.transpose(frame, axes=[2, 0, 1])
    frame = np.expand_dims(frame, axis=0)
    return frame


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


def process_output(idx_, history):
    max_hist_len = 20

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


WINDOW_NAME = 'Video Gesture Recognition'


def main():
    print("Open Camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)  # 640->width, 480->height
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    print("Build Transformer...")
    transform = get_transform()
    print("Build Executor...")
    executor, ctx = get_executor()
    buffer = (
        tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
    )
    idx = 0
    history = [2]
    history_logit = []
    idx_frame = -1

    print("Ready!")
    while True:
        idx_frame += 1
        _, img = cap.read()  # 240*320*3
        if idx_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            end = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            torch_input = img_tran.view(1, 3, img_tran.size(1), img_tran.size(2))
            img_nd = tvm.nd.array(torch_input.detach().numpy(), ctx=ctx)
            inputs = (img_nd,) + buffer
            outputs = executor(inputs)
            feat, buffer = outputs[0], outputs[1:]
            idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat.asnumpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)
            print(f"{idx_frame} {catigories[idx]}")
            current_time = time.time() - end

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
        elif key == ord('F') or key == ord('f'):
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
