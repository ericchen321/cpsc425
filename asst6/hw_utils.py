import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import numpy.linalg as LA
from PIL import Image
import random
import colorsys
import os

LOG_INTERVAL = 10
SEED = 0


def plot_loss_acc(metric_dict, figsize=(5, 15)):
    train_loss = metric_dict['train_loss']
    val_loss = metric_dict['val_loss']
    train_acc = metric_dict['train_acc']
    val_acc = metric_dict['val_acc']

    assert isinstance(train_loss, list)
    assert isinstance(val_loss, list)
    assert isinstance(train_acc, list)
    assert isinstance(val_acc, list)
    assert len(train_loss) == len(val_loss)
    assert len(train_loss) == len(train_acc)
    assert len(val_loss) == len(val_acc)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    curr_ax = ax[0]
    curr_ax.plot(list(range(len(train_loss))), train_loss, 'x-')
    curr_ax.plot(list(range(len(val_loss))), val_loss, 'o-')
    curr_ax.legend(['train_loss', 'val_loss'])
    curr_ax.set_xlabel('Epoch')
    curr_ax.set_ylabel('Loss')
    curr_ax.set_ylim([-0.5, 4])
    curr_ax.grid('on')

    curr_ax = ax[1]
    curr_ax.plot(list(range(len(train_acc))), train_acc, 'x-')
    curr_ax.plot(list(range(len(val_acc))), val_acc, 'o-')
    curr_ax.legend(['train_acc', 'val_acc'])
    curr_ax.set_xlabel('Epoch')
    curr_ax.set_ylabel('Accuracy')
    curr_ax.set_ylim([0, 105])
    curr_ax.grid('on')


def spawn_train_show(
        args, train_loader, val_loader,
        print_model=True, print_loss=True, skip_train=False):
    reset_seeds(SEED)
    model = Net(
        channel1=args['channel1'], channel2=args['channel2'],
        mid_dims=args['mid_dims'], act_name=args['act_name']
        ).to(args['device'])
    if print_model:
        print(model)
    if skip_train:
        return model
    metric_dict = train(
        args, model, args['device'], train_loader,
        val_loader, verbose=print_loss)
    best_val_acc = max(metric_dict['val_acc'])
    best_train_acc = max(metric_dict['train_acc'])
    final_val_acc = (metric_dict['val_acc'])[-1]
    final_train_acc = (metric_dict['train_acc'])[-1]
    print('The highest training accuracy is %.2f percent' % (best_train_acc))
    print('The highest validation accuracy is %.2f percent' % (best_val_acc))
    print('The final training accuracy is %.2f percent' % (final_train_acc))
    print('The final validation accuracy is %.2f percent' % (final_val_acc))
    plot_loss_acc(metric_dict)
    return model


def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fetch_data():
    reset_seeds(SEED)
    train_fraction = 0.1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform)
    test_set = datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform)
    num_trains = len(dataset)

    train_idx = list(range(int(num_trains*train_fraction)))
    val_idx = list(
        range(int(num_trains*train_fraction),
              int(num_trains*train_fraction*2)))

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(
        test_set, list(range(int(num_trains*train_fraction))))
    print("len(train_set) = %d, len(val_set) = %d, len(test_set) = %d"
            % (len(train_set), len(val_set), len(test_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=16, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=16, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=16, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


class Net(nn.Module):
    def __init__(self, act_name, channel1, channel2, mid_dims):
        super(Net, self).__init__()
        channel0 = 3
        channel3 = 64

        self.cnn = nn.Sequential(
            nn.Conv2d(channel0, channel1, 3, 1),
            get_activation(act_name),
            nn.MaxPool2d(2),
            nn.Conv2d(channel1, channel2, 3, 1),
            get_activation(act_name),
            nn.MaxPool2d(2),
            nn.Conv2d(channel2, channel3, 3, 1),
            get_activation(act_name),
        )

        mid_dims = [64*4*4] + mid_dims + [10]
        self.linear_layers = MLP(mid_dims, act_name)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
        args, model, device, train_loader,
        val_loader, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in tqdm_notebook(range(1, args['epoch'] + 1)):
        reset_seeds(SEED)
        train_epoch(
            args, model, device,
            train_loader, optimizer, epoch)
        t_loss, t_acc = test(args, model, train_loader)
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        v_loss, v_acc = test(args, model, val_loader)
        val_loss.append(v_loss)
        val_acc.append(v_acc)

        if verbose:
            print(
                'Epoch %d: training loss = %.4f, validation loss = %.4f'
                % (epoch, t_loss, v_loss))
    metric_dict = {
        'train_loss': train_loss, 'val_loss': val_loss,
        'train_acc': train_acc, 'val_acc': val_acc}
    return metric_dict


def train_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_vec = []
    running_loss = 0.0
    print_i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print_i += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if print_i % LOG_INTERVAL == 0:
            avg_loss = running_loss/LOG_INTERVAL
            loss_vec.append(avg_loss)
            running_loss = 0.0
    return loss_vec


def test(args, model, test_loader):
    device = args['device']
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    acc = 100.0*correct/len(test_loader.dataset)
    return test_loss, acc


class MLP(nn.Module):
    """
    A multi-layer perceptron with dim `dims` with activation between layers.
    """
    def __init__(self, dims, act, use_bn=False):
        torch.manual_seed(1)
        if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)

        super().__init__()
        assert isinstance(dims, list)
        assert isinstance(act, str)
        assert len(dims) > 1
        self.dims = dims
        self.use_bn = use_bn

        layer_list = []
        prev_dim = dims[0]
        dims = dims[1:]
        for i, curr_dim in enumerate(dims):
            if use_bn:
                layer_list.append(nn.BatchNorm1d(prev_dim))
            layer_list.append(nn.Linear(prev_dim, curr_dim))
            if i < len(dims) - 1:
                myact = get_activation(act)
                layer_list.append(myact)
            prev_dim = curr_dim
        self.model = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_activation(name):
    """This function return an activation constructor by name."""
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'identity':
        return Identity()
    else:
        print("Undefined activation: %s" % (name))
        assert False


def test_relu(in_shape, Layer):
    inputs = (torch.rand(tuple(in_shape)) - 0.5)*100
    inputs = inputs.detach().numpy()

    relu_np = Layer()
    relu_torch = nn.ReLU()
    out_np = relu_np.forward(inputs)
    out_torch = relu_torch(torch.FloatTensor(inputs)).detach().numpy()
    assert out_torch.shape == out_np.shape
    error = LA.norm(out_np - out_torch)
    print("norm(out_np - out_torch) = %.6f" % (error))


def test_maxpool(N, C, size, kernel_size, Layer):
    # (N, C, size, size)
    inputs = np.random.rand(N, C, size, size) * 10

    pool_np = Layer(kernel_size)
    pool_torch = nn.MaxPool2d(kernel_size)
    out_np = pool_np.forward(inputs)
    out_torch = pool_torch(torch.FloatTensor(inputs)).detach().numpy()
    assert out_torch.shape == out_np.shape
    error = LA.norm(out_np - out_torch)
    print("norm(out_np - out_torch) = %.6f" % (error))


def test_linear(N, D, K, Layer):
    linear_torch = nn.Linear(D, K)
    linear_torch.eval()
    params = list(linear_torch.parameters())
    W, b = params

    linear_np = Layer(D, K)
    linear_np.load_weights(W.detach().numpy(), b.detach().numpy())

    inputs = np.random.rand(N, D) * 10
    out_np = linear_np.forward(inputs)
    out_torch = linear_torch.forward(
            torch.FloatTensor(inputs)).detach().numpy()

    assert out_torch.shape == out_np.shape
    error = LA.norm(out_np - out_torch)
    print("norm(out_np - out_torch) = %.6f" % (error))


def test_conv2d(N, D, K, size, kernel_size, Layer):
    conv_torch = nn.Conv2d(D, K, kernel_size)
    conv_torch.eval()
    param = list(conv_torch.parameters())
    W = param[0].detach().numpy()
    b = param[1].detach().numpy()
    W = W.reshape(W.shape[0], -1)

    conv_np = Layer(D, K, kernel_size)
    conv_np.load_weights(W, b)
    inputs = np.random.rand(N, D, size, size) * 100

    out_np = conv_np.forward(inputs)
    #print(out_np)
    out_torch = conv_torch.forward(torch.FloatTensor(inputs)).detach().numpy()
    #print(out_torch)

    assert out_torch.shape == out_np.shape
    error = LA.norm(out_np - out_torch)
    print("norm(out_np - out_torch) = %.6f" % (error))


def apply_mask_rcnn(im_path, model, topk, device, figsize=(20, 20)):
    print("Visualizing %s" % (im_path))
    assert os.path.exists(im_path), "Cannot find your image (%s); make sure it is in your directory." % (im_path)
    im = Image.open(im_path)
    # resize
    width, height = im.size
    max_size = 600
    size = max(width, height)
    ratio = 1.0*max_size/size
    im = im.resize((int(width*ratio), int(height*ratio)), Image.ANTIALIAS)
    im_np = np.array(im)
    print("Resized image size: %d, %d" % (im.size[0], im.size[1]))


    # prepare inputs
    inputs = torch.FloatTensor(im_np)
    assert len(inputs.shape) == 3, "Input image should have 3 dimensions"
    assert inputs.shape[2] == 3, "Input image should have 3 channels"
    inputs = inputs.permute(2, 0, 1)
    inputs /= 255
    inputs = inputs.to(device)

    pred = model([inputs])[0]

    boxes = pred['boxes'][:topk].cpu()
    labels = pred['labels'][:topk].cpu()
    scores = pred['scores'][:topk].cpu()
    masks = pred['masks'][:topk].squeeze(1).cpu()
    masks = masks.permute(1, 2, 0)

    colors = random_colors(topk)

    im_mask = im_np
    for obj_i in range(topk):
        im_mask = apply_mask(im_mask, masks[:, :, obj_i], colors[obj_i])

    plt.figure(figsize=figsize)
    plt.imshow(im_mask)
    for box, label, score in zip(boxes, labels, scores):
        box = box.tolist()
        label_str = "%s %.2f" % (COCO_INSTANCE_CATEGORY_NAMES[label], score)
        plot_bbox(box, label_str)


def plot_bbox(xyxy, label, line='-', color='r', linewidth=2):
    """
    Plot a bounding box on a figure.
    """
    if not isinstance(xyxy, list):
        xyxy = xyxy.tolist()
    x1, y1, x2, y2 = xyxy

    top_left = [x1, y1]
    down_left = [x1, y2]
    top_right = [x2, y1]
    down_right = [x2, y2]

    plt.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]],
             linestyle=line, color=color, linewidth=linewidth)
    plt.plot([top_left[0], down_left[0]], [top_left[1], down_left[1]],
             linestyle=line, color=color, linewidth=linewidth)
    plt.plot([top_right[0], down_right[0]], [top_right[1], down_right[1]],
             linestyle=line, color=color, linewidth=linewidth)
    plt.plot([down_left[0], down_right[0]], [down_left[1], down_right[1]],
             linestyle=line, color=color, linewidth=linewidth)
    plt.text(
        x1+1, y1-3, label,
        color='white', fontsize=15,
        bbox=dict(facecolor='red', alpha=0.5))

# from PyTorch model zoo
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0.8,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
