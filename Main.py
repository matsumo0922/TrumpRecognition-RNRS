import datetime
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from tqdm import tqdm

from PrintLog import PrintLog

# ImageFolderで読み込んだ際にこの順番で並べられる
from RNRS import ResNetRs

CATS = ['10C', '10D', '10H', '10S', '11C', '11D', '11H', '11S', '12C', '12D', '12H', '12S', '13C', '13D', '13H', '13S', '1C', '1D', '1H', '1S', '2C',
        '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S',
        '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S']

# ハイパーパラメータなどの定数地
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 50
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 75
N_OUTPUT = 52


def get_device():
    """
    使用するデバイスを取得する
    GPUが2つ以上ある場合は選択させる

    :return: 使用するデバイス
    """

    if not torch.cuda.is_available():
        return torch.device("cpu")

    n_device = torch.cuda.device_count()

    if n_device == 1:
        return torch.device("cuda:0")

    devices = list(map(lambda i: f"cuda:{i[0]}", enumerate([None] * n_device)))
    device_names = list(map(lambda x: torch.cuda.get_device_name(x), devices))

    print(f"Find any GPU -> {device_names}")
    print("Please enter the index of the GPU to use (1~) > ", end="")
    index = int(input().strip())

    return torch.device(devices[index - 1])


def get_transform(is_train) -> Compose:
    """
    テンソル化などのTransformを取得する
    データ拡張などもここで行う

    Resize: リサイズ
    Normalize: 画像の正規化
    RandomErasing: ランダムで黒塗りする
    RandomHorizontalFlip: ランダムで水平反転する
    RandomVerticalFlip: ランダムで垂直反転する
    RandomApply: ランダムで以下のTransformを行う
        GaussianBlur: ぼかす
        Grayscale: グレースケール化する
        ColorJitter: 明るさ、コントラストをランダムで調整する

    :param is_train: Train用ならデータ拡張用のTransformも取得する
    :return: Torchvision.Transform.Compose
    """

    if is_train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomErasing(0.5, scale=(0.02, 0.3), ratio=(0.3, 0.3)),
            transforms.RandomHorizontalFlip(0.4),
            transforms.RandomVerticalFlip(0.4),
            transforms.RandomApply([transforms.GaussianBlur(3)], 0.2),
            # transforms.RandomApply([transforms.Grayscale()], 0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.6, contrast=0.6)], 0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])


def get_datasets():
    """
    データセットを読み込む
    ImageFolderはroot以下のディレクトリが次のような場合のみ読み込める

    root
      ├─train
      │  ├─1H
      │  │  1H-DA-1.jpg
      │  │  1H-DA-2.jpg
      │  │  1H-DA-3.jpg
      │  ├─1D
      │  │  1D-DA-1.jpg
      │  │  1D-DA-2.jpg
      │  │  1D-DA-3.jpg
      │  ├─1C
      │  │  1C-DA-1.jpg
      │  │  1C-DA-2.jpg
      │  │  1C-DA-3.jpg
      │  ├─1S

    この例の場合ではroot=root/trainに指定すれば読み込める
    ラベルは1H,1D,1Cなど画像が入っているディレクトリ名となる

    :return: Tuple[ImageFolder, ImageFolder]
    """

    train_dir = "./dataset/train"
    valid_dir = "./dataset/valid"

    train_datasets = ImageFolder(root=train_dir, transform=get_transform(True))
    valid_datasets = ImageFolder(root=valid_dir, transform=get_transform(False))

    return train_datasets, valid_datasets


def get_dataloader(train_datasets, valid_datasets, batch_size) -> Tuple[DataLoader, DataLoader]:
    """
    データローダーを取得する
    データローダーに指定している引数は以下の意味である
        batch_size: バッチサイズ, 大きすぎると学習が早くなるが局所解から抜け出せなくなり、小さすぎると学習が遅くなるが学習が過敏になる（その分過学習も起こりやすくなる）
        num_works: 並列読み込みする際に使用するCPUスレッド（かも）
        pin_memory: ページロックされたメモリ上にデータを展開, CUDAへの転送を高速化できる
        shuffle: データの読み込み順番をシャッフルする

    TODO: num_worksは使用しているCPUコア数,スレッド数を確認の上、再設定する必要あり

    :param train_datasets: Train用データセット
    :param valid_datasets: Valid用データセット
    :param batch_size: バッチサイズ
    :return: Tuple[DataLoader, DataLoader]
    """

    train_loader = DataLoader(train_datasets, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)

    return train_loader, valid_loader


def train_model(
        net: ResNetRs,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        outputs_path: str,
        logger: PrintLog
):
    """
    モデルを学習する
    CIFAR-10のサンプルとほとんど変わらない
    scalerはGPUのTensorコアを使用するためにfp16へ変換するクラス（必要なければ削除OK）
    5Epochに一回、現状のlossカーブ,acc,重みファイルを出力する

    :param net: モデル
    :param train_loader: Train用データローダー
    :param valid_loader: Valid用データローダー
    :param criterion: 損失関数
    :param optimizer: 最適化アルゴリズム
    :param device: 使用するデバイス
    :param outputs_path: lossカーブなどを出力するパス
    :param logger: 標準出力のログをとるクラス
    :return: array[array[epoch, train_loss, train_acc, valid_loss, valid_acc]]
    """

    # 学習を始めた時間
    start_time = time.perf_counter()

    # Tensorコアを使用するためのクラス(AMP)
    scaler = amp.GradScaler()

    history = np.zeros((0, 5))
    net.to(device)

    # EPOCHS回繰り返す
    for epoch in range(EPOCHS):

        train_acc, train_loss = 0.0, 0.0
        valid_acc, valid_loss = 0.0, 0.0
        num_trained, num_tested = 0.0, 0.0

        # モデルを学習モードにする（勾配自動計算モード）
        net.train()

        # DetaLoaderが返すデータ数分繰り返す
        for inputs, labels in tqdm(train_loader):
            num_trained += len(labels)

            # GPUに送る
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配初期化
            optimizer.zero_grad()

            # fp16に変換して推論,損失の計算を行い、fp32に戻す
            with amp.autocast():
                outputs = net(inputs).to(device)
                loss = criterion(outputs, labels)

            # 以下と同じ意味
            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # outputsはCATSの順で確率的な値を返す
            # この中で一番高い確率？がモデルの出力結果として扱う
            predicted = torch.max(outputs, 1)[1]

            train_loss += loss.item()
            train_acc += (predicted == labels).sum().item()

        # モデルを推論モードにする（勾配自動計算off）
        net.eval()

        for valid_inputs, valid_labels in valid_loader:
            num_tested += len(valid_labels)

            valid_inputs = valid_inputs.to(device)
            valid_labels = valid_labels.to(device)

            with amp.autocast():
                valid_outputs = net(valid_inputs).to(device)
                loss = criterion(valid_outputs, valid_labels)

            valid_predicted = torch.max(valid_outputs, 1)[1]

            valid_loss += loss.item()
            valid_acc += (valid_predicted == valid_labels).sum().item()

        # 1つ当たりのACCにする
        train_acc /= num_trained
        valid_acc /= num_tested

        # このEpochまでのlossにする
        train_loss *= (BATCH_SIZE / num_trained)
        valid_loss *= (BATCH_SIZE / num_tested)

        # このEpochが終了した時間
        end_time = time.perf_counter()

        # このEpochに費やされた時間
        elapsed_time = end_time - start_time

        # 全Epochが終了するまでに何時間かかるか取得
        eta = get_time_from_sec((elapsed_time / (epoch + 1)) * (EPOCHS - (epoch + 1)))

        message = f"Epoch [{epoch + 1}/{EPOCHS}] "
        message += f"loss: {train_loss:.5f}, acc: {train_acc:.5f}, valid loss: {valid_loss:.5f}, valid acc: {valid_acc:.5f} "
        message += f"[ETA: {str(eta[0]).zfill(2)}:{str(eta[1]).zfill(2)}:{str(eta[2]).zfill(2)}]"

        logger.println(message)

        # このEpochの結果を収める
        items = np.array([epoch, train_loss, train_acc, valid_loss, valid_acc])
        history = np.vstack((history, items))

        # 5Epochに一回、現状のlossカーブ,acc,重みファイルを出力する
        if (epoch + 1) % 5 == 0 and (epoch + 1) != EPOCHS:
            save_weight(outputs_path, epoch, net)
            show_loss_carve(outputs_path, epoch, history)
            show_accuracy_graph(outputs_path, epoch, history)

    return history


def show_loss_carve(parent_path, epoch, history: numpy.ndarray):
    """
    Lossカーブを出力する
    指定されたパスに画像を保存する

    :param parent_path: 保存したいディレクトリのパス
    :param epoch: 現時点でのEpoch数
    :param history: train_model()で返される2次元配列
    """

    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(history[:, 0], history[:, 1], "b", label="train")
    plt.plot(history[:, 0], history[:, 3], "k", label="valid")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("loss carve")
    plt.legend()

    plt.savefig(f"{parent_path}/loss-epoch-{epoch}.jpg")
    plt.show()


def show_accuracy_graph(parent_path, epoch, history: numpy.ndarray):
    """
    ACCカーブを出力する
    指定されたパスに画像を保存する

    :param parent_path: 保存したいディレクトリのパス
    :param epoch: 現時点でのEpoch数
    :param history: train_model()で返される2次元配列
    """

    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(history[:, 0], history[:, 2], "b", label="train")
    plt.plot(history[:, 0], history[:, 4], "k", label="valid")
    plt.xlabel("iter")
    plt.ylabel("acc")
    plt.title("accuracy")
    plt.legend()

    plt.savefig(f"{parent_path}/accuracy-epoch-{epoch}.jpg")
    plt.show()


def show_result(net: ResNetRs, valid_loader: DataLoader, device, parent_path):
    """
    ValidLoaderから50件読み込み、推論した結果を出力し、指定されたパスに保存する
    [正解ラベル]:[推論ラベル]という形式で出力され、間違っている場合は青で表示される

    :param net: 学習済みのモデル
    :param valid_loader: Valid用のデータローダー
    :param device: 使用するデバイス
    :param parent_path: 保存したいディレクトリのパス
    """

    for images, labels in valid_loader:
        break

    net.to(device)
    net.eval()

    test_labels = labels.to(device)
    test_images = images.to(device)

    outputs = net(test_images).to(device)
    predicts = torch.max(outputs, 1)[1]

    plt.figure(figsize=(21, 15))

    for index in range(50):
        ax = plt.subplot(5, 10, index + 1)

        answer_label = CATS[test_labels[index].item()]
        predicted_label = CATS[predicts[index].item()]

        color = "k" if answer_label == predicted_label else "b"
        ax.set_title(f"{answer_label}:{predicted_label}", c=color, fontsize=20)

        image_np = images[index].numpy().copy()
        image = np.transpose(image_np, (1, 2, 0))
        image = (image + 1) / 2

        plt.imshow(image)
        ax.set_axis_off()

    plt.savefig(f"{parent_path}/result.jpg")
    plt.show()


def save_weight(parent_path, epoch, model):
    """
    モデルの重みファイルを出力する
    指定されたパスに重みファイルを保存する

    :param parent_path: 保存したいディレクトリのパス
    :param epoch: 現時点でのEpoch数
    :param model: モデル
    """

    save_dir = f"{parent_path}/weights"
    save_path = f"{save_dir}/weight-epoch-{epoch}.pth"

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def get_time():
    """
    現在時刻を取得する
    2011-6-23-10-8-24のような形で返される

    :return: %Y-%m-%d-%H-%M-%S形式のstr
    """

    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_time_from_sec(sec) -> Tuple[int, int, int]:
    """
    秒数を時間,分,秒の形で取得する

    :param sec: 秒数
    :return: Tuple[時間, 分, 秒]
    """

    timedelta = datetime.timedelta(seconds=sec)
    m, s = divmod(timedelta.seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def get_predict(path: str, net: ResNetRs, device):
    """
    学習済みモデルを使用して任意の画像を推論する

    :param path: 任意の画像パス, jpg,jpeg,png,webpなどの拡張子に対応, heicは非対応
    :param net: 学習済みのモデル
    :param device: 使用するデバイス
    :return: 確率が高い順にソートした推論結果, array[int]
    """

    transform = get_transform(False)

    image = Image.open(path)
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    net.to(device)
    net.eval()

    output = net(image)
    output = output.tolist()[0]
    output = list(enumerate(output))
    output = sorted(output, key=lambda x: x[1], reverse=True)

    return output


def train():
    """
    学習を実行する
    最後にlossカーブ,accカーブ,50件の推論結果,重みファイルを出力する
    """

    # 現在時刻を取得する
    time_id = get_time()

    # 現在時刻をディレクトリ名にしたディレクトリを生成する, ここにlossカーブなどのoutputを出力する
    result_path = f"./results/{time_id}"
    os.makedirs(result_path, exist_ok=True)

    # 標準出力のログをとるクラスをインスタンス化する
    log = PrintLog(result_path + "/log.txt")

    # 使用するデバイスを取得する
    device = get_device()

    # データセット,データローダーを取得する
    train_datasets, valid_datasets = get_datasets()
    train_loader, valid_loader = get_dataloader(train_datasets, valid_datasets, BATCH_SIZE)

    # 読み込まれたデータ数を表示
    log.println(f"train data: {len(train_datasets)}, valid data: {len(valid_datasets)}")

    # モデル,損失関数,最適化アルゴリズムをインスタンス化する
    net = ResNetRs(N_OUTPUT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # モデルの学習を実行する
    history = train_model(net, train_loader, valid_loader, criterion, optimizer, device, result_path, log)

    # lossカーブ,accカーブ,50件の推論結果,重みファイルを出力する
    save_weight(result_path, "finish", net)
    show_loss_carve(result_path, "finish", history)
    show_accuracy_graph(result_path, "finish", history)
    show_result(net, valid_loader, device, result_path)


def predict():
    """
    重みファイルを読み込み、学習済みモデルを再構築して任意の画像を推論する
    最初に重みファイルを読み込む, 以降は任意の画像のパスを入力する
    出力は以下のようになる

    [推論ラベル], [確率], [実際の値]
    """

    # 重みファイルのパスを取得する
    print("Enter the path of the PTH file > ", end="")
    pth_path = input().strip()

    # 重みファイルを読み込み、学習済みのモデルを再構築する
    device = get_device()
    net = ResNetRs(N_OUTPUT)
    net.load_state_dict(torch.load(pth_path))

    # 無限に繰り返す
    while True:

        # 推論したい任意の画像のパスを取得する
        print("Enter the path of the image to predict > ", end="")
        image_path = input().strip()

        # exitと入力されたら無限ループから抜ける
        if image_path.lower() == "exit":
            print("Ended the process.")
            break

        # 推論して結果を出力する
        result = get_predict(image_path, net, device)
        rate_sum = sum(list(map(lambda x: x[1], result)))
        take_list = list(map(lambda x: (CATS[x[0]], x[1]), result[:7]))

        for trump, rate in take_list:
            print(f"{trump}, {(rate / rate_sum) * 100}%, {rate}")


def info():
    """
    モデルの構造、全結合層に入力されるノード数を出力する
    """

    # モデルを取得する
    device = get_device()
    net = ResNetRs(N_OUTPUT).to(device)

    # 入力画像を模した疑似テンソルを構築する
    tensor = torch.FloatTensor(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)

    # 全結合層に入力されるノード数を出力する
    print(net(tensor).to(device).shape)

    # モデルの構造を出力する
    print(summary(model=net, input_size=(BATCH_SIZE, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])))


def main():
    """
    このプログラムのエントリポイント
    推論モード, 学習モード, 情報モードを選択する
    """

    print("Choose mode predict mode [P], train mode [T] or info mode [I]: > ", end="")
    mode = input().strip()

    if mode.upper() == "P":
        predict()
    elif mode.upper() == "T":
        train()
    elif mode.upper() == "I":
        info()
    else:
        print("Invalid input. ")


if __name__ == "__main__":
    main()
