"""
Uczenie sieci z highway_network.py na CIFAR-10 i pomiar wyniku.

Wykład 1 definiuje uczenie trzema rzeczami, które tu są ustalone z góry.
Zadanie: obrazkowi 32x32 z trzech kolorów przypisać jedną z 10 klas.
Doświadczenie: stały zbiór CIFAR-10. Uczenie jest wsadowe, model
nie doucza się później na nowych zdjęciach.
Miara: dokładność, precision, recall, F1 i macierz pomyłek.
Miara jest umową sprzed treningu. Optymalizator jej nie widzi.

Optymalizator widzi stratę: entropię krzyżową z wykładu 2.
To średni błąd na danych, które mamy (ryzyko empiryczne).
Chcielibyśmy małego błędu na nowych danych (ryzyko).
Różnica między nimi to luka generalizacji.
Dlatego są trzy zbiory, a nie jeden.

Zbiór treningowy dopasowuje wagi.
Zbiór walidacyjny porównuje epoki i ustawienia.
Zbiór testowy jest ruszany raz, na końcu.
Błąd treningowy po wybraniu najlepszych wag jest zbyt optymistyczny:
minimalizacja sama z siebie zaniża błąd na danych, po których schodziła.

Pętla uczenia jest szkieletem z wykładu 2, tym samym do końca kursu:
forward (policz wynik) -> loss (policz błąd) -> backward (policz nachylenie)
-> step (krok wag w stronę mniejszego błędu).
"""

import argparse
import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

from highway_network import (
    DeepFCHighwayNet,
    DeepFCPlainNet,
    build_highway11_cifar10,
    build_highway19_cifar10,
    count_parameters,
)


class TensorCIFAR10Dataset(Dataset):
    """
    Gotowe obrazki i etykiety, już po podziale i standaryzacji.

    Transformacja jest osobno, bo losowy kadr i odbicie
    mają dotyczyć tylko treningu. Walidacja i test dostają None:
    pomiar ma być na stałym obrazku, nie na losowej wersji.
    """

    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        # Liczba przykładów. Wykład 1 oznacza ją przez m.
        return len(self.targets)

    def __getitem__(self, idx):
        # Jeden przykład: obraz x i etykieta y.
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def get_cifar10_loaders(data_dir="./data", batch_size=128, val_fraction=0.1, num_workers=0, seed=42):
    """
    Kontrakt danych z wykładu 1.

    Oficjalny CIFAR-10 ma 50 000 obrazów uczących i 10 000 testowych,
    po równo w 10 klasach. Testu nie dzielimy.
    Z 50 000 odcinamy 10% na walidację: wychodzi 45 000 / 5 000 / 10 000.

    Kolejność jest częścią kontraktu:
    najpierw podział, potem średnia i odchylenie tylko z treningu,
    potem model. Wszystko, czego model uczy się z danych,
    wolno liczyć wyłącznie na części treningowej.
    Inaczej test wnosi swoje statystyki do treningu
    i wynik testowy przestaje być pomiarem na danych nieoglądanych.
    Wykład nazywa to wyciekiem przez preprocessing.
    """
    # Pobierz zbiór tylko wtedy, gdy nie leży już na dysku.
    already_extracted = os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py"))
    raw_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=not already_extracted)
    raw_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=not already_extracted)

    # Indeksy 50 000 obrazów uczących. Test tu nie wchodzi.
    indices = np.arange(len(raw_train.targets))
    # Podział stratyfikowany: w treningu i w walidacji ten sam udział klas.
    # Wykład 1 pokazuje na małej próbce, że zwykły losowy podział
    # mocno buja udziałem klasy. Przy równych klasach CIFAR-10
    # stratyfikacja daje dokładnie 4500 i 500 obrazów na klasę.
    # Ziarno 42 odtwarza ten sam podział. Na wykładzie 2
    # to samo ziarno odtwarza wynik regresji.
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=raw_train.targets,
        random_state=seed,
    )

    # Obraz z kształtu (wysokość, szerokość, kolory) na (kolory, wysokość, szerokość).
    # Piksel 0..255 ląduje w [0, 1]. To jeszcze nie standaryzacja.
    x_train_raw = torch.from_numpy(raw_train.data[train_idx]).permute(0, 3, 1, 2).float().div_(255.0)
    y_train = torch.tensor(np.array(raw_train.targets)[train_idx], dtype=torch.long)

    x_val_raw = torch.from_numpy(raw_train.data[val_idx]).permute(0, 3, 1, 2).float().div_(255.0)
    y_val = torch.tensor(np.array(raw_train.targets)[val_idx], dtype=torch.long)

    x_test_raw = torch.from_numpy(raw_test.data).permute(0, 3, 1, 2).float().div_(255.0)
    y_test = torch.tensor(np.array(raw_test.targets), dtype=torch.long)

    # Średnia i odchylenie osobno dla każdego z trzech kolorów,
    # liczone po obrazkach oraz po wysokości i szerokości.
    # Tylko z treningu. Walidacja i test dostaną te same trzy liczby.
    train_mean = x_train_raw.mean(dim=(0, 2, 3), keepdim=True)
    train_std = x_train_raw.std(dim=(0, 2, 3), keepdim=True)

    # x' = (x - średnia_treningu) / odchylenie_treningu.
    # Po tym kanały treningu mają średnią 0 i odchylenie 1.
    # Wykład 2 liczy to ręcznie na Irisie: ten sam krok uczenia
    # na surowych cechach rozjeżdża błąd, a po standaryzacji schodzi.
    # Jeden krok ma działać we wszystkich kierunkach naraz,
    # a cechy o różnej skali wyginają dolinę błędu nierówno.
    x_train = (x_train_raw - train_mean) / train_std
    x_val = (x_val_raw - train_mean) / train_std
    x_test = (x_test_raw - train_mean) / train_std

    # Augmentacja jest w księdze z wykładu 1: więcej różnorodnych
    # widoków tych samych danych, mniejsza skłonność do zapamiętania
    # szumu jednego kadru. Koszt to lekkie pogorszenie dopasowania
    # do oryginalnego kadru. Wyprowadzenie jest na wykładzie 7.
    # Tylko trening. Walidacja i test oglądają obraz taki, jaki jest.
    train_transform = transforms.Compose(
        [
            # Ramka 4 pikseli i losowy kadr 32x32.
            # Ramka jest z zer. Po standaryzacji zero to nie średnia
            # jasność, tylko ciemny piksel. Brzeg kadru niesie stały sygnał.
            transforms.RandomCrop(32, padding=4),
            # Odbicie lewo-prawo. Dla klas CIFAR-10 orientacja nie jest etykietą.
            transforms.RandomHorizontalFlip(),
        ]
    )

    train_set = TensorCIFAR10Dataset(x_train, y_train, transform=train_transform)
    val_set = TensorCIFAR10Dataset(x_val, y_val, transform=None)
    test_set = TensorCIFAR10Dataset(x_test, y_test, transform=None)

    # pin_memory przyspiesza kopiowanie paczki na kartę, gdy karta jest.
    pin_memory = torch.cuda.is_available()
    # Trwałe procesy ładowania mają sens tylko wtedy, gdy w ogóle są.
    persistent = num_workers > 0

    # Paczki po 128. To mini-batch z wykładu 2:
    # nie cały zbiór naraz (dokładny, drogi gradient)
    # i nie jeden obrazek (gradient nieobciążony, ale bardzo szumny).
    # Szum paczki potrafi wyprowadzić uczenie ze słabego miejsca.
    # Mieszanie dotyczy tylko treningu. Walidacja i test mają stałą kolejność,
    # bo i tak liczymy średnią po wszystkich przykładach.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader


def compute_dummy_baseline(train_loader, val_loader, num_classes=10):
    """
    Głupi klasyfikator, zanim jakakolwiek waga dostanie kredyt.

    Wykład 1: zanim uzna się wynik, liczy się baseline.
    Model, który go nie przebija, nie nauczył się nic ponad rozkład klas.

    Na CIFAR-10 klasy są równe, więc klasyfikator „zawsze ta sama klasa”
    ma dokładność 10%. Przy danych nierównych dokładność potrafi
    wyglądać świetnie i być bezużyteczna: wykład daje przykład
    „wszyscy zdrowi” przy 1% chorych, accuracy 99%, zero wykrytych chorób.
    Tu tej pułapki nie ma, a próg i tak drukujemy.

    Dwie liczby dummy nie pochodzą z tego samego głupiego modelu.
    Dokładność i F1: zawsze klasa najczęstsza w treningu.
    Strata: model, który każdej klasie daje prawdopodobieństwo 1/10.
    Strata takiego modelu to ln(10) ≈ 2,30.
    Model, który stawia prawdopodobieństwo 1 na jednej klasie,
    ma na błędach stratę nieskończoną. Tego tu nie liczymy.
    """
    y_train = train_loader.dataset.targets.numpy()
    y_val = val_loader.dataset.targets.numpy()
    # Ile razy każda klasa występuje w treningu.
    counts = np.bincount(y_train, minlength=num_classes)
    # Najczęstsza klasa. Przy remisie wygrywa mniejszy numer.
    majority_class = int(np.argmax(counts))
    # Cała walidacja opisana tą jedną etykietą.
    dummy_preds = np.full_like(y_val, majority_class)
    dummy_acc = 100.0 * float((dummy_preds == y_val).mean())
    # -ln(1/K) = ln(K). Średnia strata, gdy softmax jest płaski.
    dummy_loss = math.log(num_classes)
    # F1 macro: dla każdej klasy osobno, potem zwykła średnia.
    # Klasa wybrana ma wysoką czułość i niską precyzję.
    # Pozostałe dziewięć mają zero trafień, więc ich F1 to 0.
    # Średnia wychodzi koło 1,8%. Dopiero wynik wyraźnie wyżej
    # znaczy, że sieć rozróżnia klasy, a nie tylko trafia w jedną.
    dummy_f1 = 100.0 * float(f1_score(y_val, dummy_preds, average="macro", zero_division=0))
    return {
        "majority_class": majority_class,
        "val_loss": dummy_loss,
        "val_acc": dummy_acc,
        "val_macro_f1": dummy_f1,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Jedno przejście po całym zbiorze treningowym.

    Wykład 2 nazywa epochem jedno przejście po danych.
    Przy 45 000 obrazach i paczce 128 to około 352 kroków.
    Ostatnia paczka jest krótsza, bo 45 000 nie dzieli się przez 128.

    Szkielet kroku, nazwany na wykładzie na stałe:
    forward -> loss -> backward -> step.
    Dla Adaline każda z tych rzeczy jest jedną linijką algebry.
    Tu backward liczy automatyczne różniczkowanie.
    Kierunek kroku jest minus gradient: idziemy w stronę,
    w której błąd maleje najszybciej.
    """
    # Tryb treningu: dropout zeruje kanały, batch norm bierze statystyki paczki.
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        # Paczka na kartę albo na procesor. non_blocking ma sens przy pin_memory:
        # kopiowanie może iść w tle, gdy poprzedni krok jeszcze liczy.
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # PyTorch dodaje gradienty do już zapisanych.
        # Bez zerowania krok widziałby sumę z poprzednich paczek.
        optimizer.zero_grad()
        # Forward. Sieć zwraca 10 logitów, nie klasę i nie prawdopodobieństwo.
        outputs = model(inputs)
        # Średnia strata na tej paczce. To kawałek ryzyka empirycznego.
        loss = criterion(outputs, targets)
        # Nachylenie straty względem każdej wagi.
        loss.backward()
        # Krok: wagi <- wagi - krok * nachylenie, plus momentum i kara na wagi.
        optimizer.step()

        # Wagę paczki bierzemy z liczby obrazków, nie „jedna paczka = jeden głos”.
        # Krótka ostatnia paczka nie waży tyle, co pełna.
        # Średnia epoki jest wtedy średnią po przykładach, tak jak (1/m) * suma strat.
        running_loss += loss.item() * inputs.size(0)
        # Klasa o największym logici. Ta sama klasa wygrałaby na prawdopodobieństwach,
        # bo exp zachowuje kolejność. Do decyzji softmax nie jest potrzebny.
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Dokładność jest tylko do wykresu. Optymalizator jej nie widział.
    # Dokładność ma pochodną prawie wszędzie zero: małe ruszenie wagi
    # zwykle nie zmienia żadnej decyzji. Wykład 2 dlatego porzuca próg
    # perceptronu na czas uczenia i zostawia gładką stratę.
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_detailed=False):
    """
    Pomiar na danych, których krok uczenia w tym wywołaniu nie rusza.

    no_grad: nie zapisujemy nachylenia. To nie jest trening.
    model.eval() wyłącza dropout i przełącza batch norm na zapamiętane średnie.

    W pętli epok ta funkcja idzie po walidacji.
    Po wybraniu najlepszej epoki idzie raz po teście.

    Krótki wynik to strata i dokładność, do krzywej.
    Pełny wynik to reszta miar z wykładu 1.
    Strata i miara to dwie różne liczby.
    Strata musi mieć pochodną, bo po niej schodzimy.
    Miara mówi, czy zadanie jest zrobione.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        # Metryki liczymy na procesorze, już po decyzji.
        all_preds.append(predicted.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    total = len(y_true)
    avg_loss = running_loss / total
    # Wektor zer i jedynek: 1 tam, gdzie klasa się zgadza.
    # Średnia tego wektora to dokładność w skali 0–1.
    correct_vec = (y_pred == y_true).astype(np.float64)
    acc = 100.0 * float(correct_vec.mean())

    if not return_detailed:
        return avg_loss, acc

    # Dla K klas każda klasa po kolei gra rolę „tej szukanej”,
    # reszta zlewa się w „nie tę”.
    # precision: z tego, co sieć wskazała jako klasę, ile było trafione.
    # recall: z prawdziwych przykładów klasy, ile sieć znalazła.
    # F1: średnia harmoniczna tych dwóch. Wysoka tylko wtedy, gdy obie są wysokie.
    # macro: zwykła średnia po 10 klasach, każda klasa z wagą 1.
    # Micro zlewa wszystkie komórki w jeden licznik i przy jednej etykiecie
    # na obrazek schodzi się z dokładnością. Dlatego raportujemy macro:
    # dokładność już jest wersją micro.
    # Na zrównoważonym teście macro recall też schodzi się z dokładnością,
    # bo każda klasa ma tyle samo zdjęć. Macro precision już nie musi:
    # mianownikiem jest liczba wskazań danej klasy, a ta bywa różna.
    macro_prec = 100.0 * float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = 100.0 * float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = 100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    # Macierz 10x10. Wykład 1 rysuje tabelę 2x2: trafienia i dwa rodzaje pomyłek.
    # Przy 10 klasach ta tabela urasta. Wiersz to prawda, kolumna to decyzja.
    # Dokładność składa całą tabelę w jedną liczbę i gubi, które klasy się mylą.
    cm = confusion_matrix(y_true, y_pred)
    # Jeden podział daje jedną liczbę, a ta liczba się waha.
    # Wykład 1 podaje wynik mieszkaniowy z 95-procentowym przedziałem.
    # Tu to samo na wektorze trafień: średnia ± współczynnik t * błąd średniej.
    # Przy 10 000 obrazów testu współczynnik t jest praktycznie 1,96.
    # Przedział mówi o szumie pomiaru tego modelu, nie o szumie szukania architektury.
    ci_low, ci_high = stats.t.interval(
        0.95, len(correct_vec) - 1, loc=correct_vec.mean(), scale=stats.sem(correct_vec)
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "ci_95": (100.0 * float(ci_low), 100.0 * float(ci_high)),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def analyze_highway_gates(model, loader, device, save_path="highway_gates_analysis.png"):
    """
    Po treningu: czy bramki naprawdę się różnią.

    Wiersz wykresu to głębokość warstwy, kolumna to kanał.
    Ujemny bias bramki znaczy, że sieć nie otworzyła jej na oścież.
    Niskie średnie t znaczy, że warstwa głównie przepuszcza sygnał.
    Wyższe t znaczy, że warstwa naprawdę przekształca cechy.
    Dla jednego obrazka bramka zależy od tego obrazka:
    t jest funkcją x, nie globalnym przełącznikiem.

    Sieć płaska nie ma bramek, więc funkcja kończy się od razu.
    """
    model.eval()
    hw_layers = model.get_highway_layers()
    if not hw_layers:
        return

    num_layers = len(hw_layers)
    # Suma bramek po obrazkach, osobno na warstwę. Średnią policzymy na końcu.
    gate_sums = [0.0 for _ in range(num_layers)]
    total_samples = 0
    single_gate = None
    single_out = None

    for batch_idx, (inputs, _) in enumerate(loader):
        # Etykiety tu nie są potrzebne. Oglądamy bramki, nie błąd.
        inputs = inputs.to(device)
        # Forward w trybie eval zapisuje last_gate i last_output w każdej warstwie.
        _ = model(inputs)
        batch_size = inputs.size(0)
        total_samples += batch_size

        for i, layer in enumerate(hw_layers):
            g = layer.last_gate
            # Bramka splotowa ma kształt obrazka. Uśredniamy po pikselach,
            # zostaje jedna liczba na kanał, na każdy obrazek w paczce.
            if g.dim() == 4:
                g_mean_spatial = g.mean(dim=(2, 3))
            else:
                g_mean_spatial = g
            # Suma po obrazkach z paczki. Zostaje wektor kanałów.
            gate_sums[i] = gate_sums[i] + g_mean_spatial.sum(dim=0).cpu().numpy()

        # Pierwsza paczka: jeden konkretny obrazek, bez uśredniania po danych.
        if batch_idx == 0:
            single_gate = []
            single_out = []
            for layer in hw_layers:
                g0 = layer.last_gate[0]
                o0 = layer.last_output[0]
                if g0.dim() == 3:
                    g0 = g0.mean(dim=(1, 2))
                    o0 = o0.mean(dim=(1, 2))
                single_gate.append(g0.cpu().numpy())
                single_out.append(o0.cpu().numpy())

        # Około 2000 obrazków wystarcza do średniej bramki.
        # Cały test by tu nic nowego nie powiedział, a liczyłby dłużej.
        if total_samples >= 2000:
            break

    mean_gates = [g / total_samples for g in gate_sums]
    # Bias bramki T po treningu. To parametr, nie wyjście sigmoidu.
    biases = [layer.T.bias.detach().cpu().numpy() for layer in hw_layers]

    # Warstwy mogą mieć różną liczbę kanałów. Macierz wyrównujemy NaN-ami,
    # żeby dało się ją narysować jednym obrazkiem.
    max_blocks = max(len(b) for b in biases)

    def pad_to_matrix(arr_list):
        mat = np.full((len(arr_list), max_blocks), np.nan)
        for idx, arr in enumerate(arr_list):
            mat[idx, : len(arr)] = arr
        return mat

    bias_mat = pad_to_matrix(biases)
    mean_gate_mat = pad_to_matrix(mean_gates)
    single_gate_mat = pad_to_matrix(single_gate)
    single_out_mat = pad_to_matrix(single_out)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    titles = [
        "Transform Gate Biases",
        "Mean Transform Gate Outputs",
        "Single Sample Gate Outputs",
        "Single Sample Block Outputs",
    ]
    mats = [bias_mat, mean_gate_mat, single_gate_mat, single_out_mat]
    cmaps = ["viridis", "magma", "magma", "coolwarm"]

    for ax, title, mat, cmap in zip(axes, titles, mats, cmaps):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Channel / Block")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Highway Layer Depth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_history(history, dummy_acc=10.0, save_path="training_curves.png"):
    """
    Dwie krzywe z wykładu 1, czytane po treningu.

    Lewa: strata na treningu i na walidacji.
    Prawa: dokładność, z linią głupiego klasyfikatora.

    Jak czytać, tabela z wykładu:
    oba błędy wysokie i blisko siebie — rodzina za mała (duży bias),
    trening dużo lepszy niż walidacja — model zapamiętuje szum (duża wariancja),
    oba niskie i blisko siebie — przy tej pojemności można przestać kręcić.

    Rozkład błędu na bias, wariancję i szum wykład wyprowadza dla błędu kwadratowego.
    Tu etykiety są czystymi klasami, więc składnika „szum pomiaru” nie estymujemy.
    Tabela diagnostyczna i tak się czyta.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Loss Curve (Train vs Validation)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Validation Accuracy")
    # Próg, od którego dokładność w ogóle coś znaczy. Na CIFAR-10 to 10%.
    ax2.axhline(dummy_acc, color="gray", linestyle=":", linewidth=1.2, label=f"Dummy Baseline ({dummy_acc:.1f}%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curve (Train vs Validation)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def build_model(arch, transform_bias, use_bn, fc_layers=19):
    """
    Wybór rodziny funkcji. To pokrętło pojemności z wykładu 1.

    Na wykładzie pokrętłem jest stopień wielomianu.
    Tu jest nim architektura i liczba warstw.
    highway11 i highway19 to sieci splotowe.
    fc_highway to głęboka sieć w pełni połączona z bramkami.
    fc_plain to ten sam stos bez bramek, do porównania.
    """
    if arch == "highway11":
        return build_highway11_cifar10(transform_bias=transform_bias, use_bn=use_bn)
    if arch == "highway19":
        return build_highway19_cifar10(transform_bias=transform_bias, use_bn=use_bn)
    if arch == "fc_highway":
        # 3072 = 32*32*3. Obraz wchodzi jako płaski wektor.
        return DeepFCHighwayNet(
            input_dim=3072, hidden_dim=50, num_highway_layers=fc_layers, num_classes=10, transform_bias=transform_bias
        )
    if arch == "fc_plain":
        return DeepFCPlainNet(input_dim=3072, hidden_dim=71, num_hidden_layers=fc_layers, num_classes=10)
    raise ValueError(f"Unknown architecture: {arch}")


def main():
    """
    Jeden przebieg: dane, baseline, uczenie, wybór epoki, jeden test.

    Hiperparametry wolno wybierać na walidacji. Na teście nie.
    Każde porównanie zużywa część wartości walidacji,
    bo wybrany model dopasowuje się do tej konkretnej próbki.
    Test ruszony drugi raz stałby się po cichu kolejną walidacją.
    """
    parser = argparse.ArgumentParser()
    # Rodzina funkcji. Domyślnie mniejsza sieć splotowa.
    parser.add_argument(
        "--arch", type=str, default="highway11", choices=["highway11", "highway19", "fc_highway", "fc_plain"]
    )
    parser.add_argument("--epochs", type=int, default=15)
    # Rozmiar paczki. Mini-batch z wykładu 2.
    parser.add_argument("--batch-size", type=int, default=128)
    # Krok uczenia eta. Wykład 2: za duży rozjeżdża stratę, za mały pełznie.
    # 0,05 stoi na danych już wystandaryzowanych.
    parser.add_argument("--lr", type=float, default=0.05)
    # Momentum nie jest w wyprowadzeniu z wykładu 2.
    # Czysty krok to wagi minus eta razy gradient.
    # Tu krok pamięta poprzedni kierunek. 0,9 to siła tej pamięci.
    parser.add_argument("--momentum", type=float, default=0.9)
    # Kara na duże wagi. Wykład 2 wyprowadza ją jako regresję grzbietową (ridge):
    # do błędu dopisujemy lambda razy suma kwadratów wag.
    # W głębokich sieciach ta kara wraca pod nazwą weight decay.
    # Lambda jest hiperparametrem i wolno ją wybrać na walidacji, nigdy na teście.
    # Wykład każe szukać jej na skali logarytmicznej. 5e-4 leży na tej osi.
    # W księdze bias–wariancja: trochę gorsze dopasowanie do treningu,
    # mniejsza skłonność do szumu.
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    # Startowy bias bramki. Patrz HighwayLinear: sigmoid(-2) ≈ 0,12.
    parser.add_argument("--transform-bias", type=float, default=-2.0)
    # 10% z 50 000 idzie na walidację. Przy 50 000 przykładów wykład
    # pozwala zastąpić k-krotną walidację krzyżową jednym podziałem.
    # K-fold jest lekarstwem na szum wyniku przy małym zbiorze.
    parser.add_argument("--val-fraction", type=float, default=0.1)
    # Liczba warstw ukrytych w sieciach w pełni połączonych.
    parser.add_argument("--fc-layers", type=int, default=19)
    # Wyłącza batch norm. Domyślnie jest włączona.
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="./data")
    # Tu lądują wagi z epoki o najlepszej walidacji, a na końcu też liczby z testu.
    parser.add_argument("--save-path", type=str, default="best_highway_cifar10.pt")
    args = parser.parse_args()

    # To samo ziarno co przy podziale danych: start wag też da się powtórzyć.
    torch.manual_seed(42)
    np.random.seed(42)
    # Przy stałym rozmiarze obrazka cuDNN dobiera szybki wariant splotu.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Karta, jeśli jest. Inaczej procesor. Wzory się nie zmieniają.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )
    print(
        f"Split contract: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val (stratified) / "
        f"{len(test_loader.dataset)} test"
    )

    # Baseline zanim powstanie sieć. Dopiero wynik powyżej niego coś znaczy.
    dummy = compute_dummy_baseline(train_loader, val_loader)
    print(
        f"Dummy baseline (majority class={dummy['majority_class']}): "
        f"Val Loss: {dummy['val_loss']:.4f} | Val Acc: {dummy['val_acc']:.2f}% | "
        f"Val Macro F1: {dummy['val_macro_f1']:.2f}%"
    )

    model = build_model(
        args.arch, transform_bias=args.transform_bias, use_bn=not args.no_bn, fc_layers=args.fc_layers
    ).to(device)
    # Liczba wag to rozmiar theta. Sama liczba nie mówi, czy rodzina jest za duża.
    # To mówią krzywe treningu i walidacji.
    print(f"Architecture: {args.arch} | Parameters: {count_parameters(model):,} | Transform Bias: {args.transform_bias}")

    # Entropia krzyżowa dla 10 klas. Wykład 2 wyprowadza ją z wiarygodności,
    # nie jako wzór wybrany z kapelusza.
    # Sieć daje logity z. Softmax zamienia je na prawdopodobieństwa,
    # dodatnie i sumujące się do 1.
    # Dla prawdziwej klasy c strata przykładu to -ln(p_c).
    # Im pewniejsza poprawna klasa, tym mniejsza strata.
    # CrossEntropyLoss dostaje logity, nie gotowe prawdopodobieństwa:
    # sam liczy log-softmax i odejmuje maksimum przed exp,
    # bo exp dużych liczb wylewa float.
    # Przesunięcie wszystkich logitów o stałą nie zmienia softmaxu.
    # Błąd kwadratowy byłby tu złą stratą: pasuje do szumu gaussowskiego
    # na liczbie, a etykieta klasy gaussowska nie jest.
    # Gradient tej straty po logitach to „prawdopodobieństwo minus etykieta”.
    # Od tej różnicy backward zaczyna łańcuch w głąb sieci.
    criterion = nn.CrossEntropyLoss()
    # SGD: krok w stronę minus gradientu, z pamięcią kierunku i karą na wagi.
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    # Wykład 2 ma stały krok. Tu krok maleje kosinusem przez wszystkie epoki.
    # Pod koniec mniejszy krok siada bliżej minimum.
    # Na starcie większy krok szybciej schodzi z wysokiej straty.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        # Wagi rusza tylko ta linia. Walidacja poniżej niczego nie aktualizuje.
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Walidacja bez augmentacji, w trybie eval. Służy do wyboru epoki.
        val_metrics = evaluate(model, val_loader, criterion, device, return_detailed=True)
        # Po epoce krok uczenia maleje zgodnie z kosinusem.
        scheduler.step()
        elapsed = time.time() - start_t

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Zapis wag z epoki o najwyższej dokładności walidacyjnej.
        # To praktyczna forma early stopping z księgi wykładu 1:
        # dalsze epoki wolno policzyć, ale do testu wchodzą wagi
        # z momentu, w którym walidacja była najlepsza,
        # nie wagi z ostatniego kroku.
        # Sam wybór epoki zużywa walidację, więc ta dokładność
        # jest odrobinę optymistyczna. Uczciwy pomiar jest na teście.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "val_macro_f1": val_f1,
                    "transform_bias": args.transform_bias,
                },
                args.save_path,
            )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | "
            f"Best Val: {best_val_acc:.2f}% | Time: {elapsed:.1f}s"
        )

    # Krzywe do diagnozy bias / wariancja. Linia dummy na wykresie dokładności.
    plot_training_history(history, dummy_acc=dummy["val_acc"], save_path="training_curves.png")

    # Wracamy do wag z najlepszej epoki. Ostatnia epoka nie musi nią być.
    best_ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Jedno dotknięcie testu, po wszystkich decyzjach.
    # Zgodność walidacji z testem jest tym, po co trzeci zbiór leży w spokoju.
    # Na wykładzie 1 błąd z walidacji krzyżowej i błąd testu były blisko siebie:
    # procedura przewidziała własny błąd poza próbą.
    test_metrics = evaluate(model, test_loader, criterion, device, return_detailed=True)
    ci_lo, ci_hi = test_metrics["ci_95"]

    # Dopisujemy liczby testowe do tego samego pliku wag.
    # Drugi trening ich nie używa. To zapis pomiaru, nie nowe uczenie.
    best_ckpt["test_acc"] = test_metrics["acc"]
    best_ckpt["test_loss"] = test_metrics["loss"]
    best_ckpt["test_ci_95"] = test_metrics["ci_95"]
    best_ckpt["test_macro_precision"] = test_metrics["macro_precision"]
    best_ckpt["test_macro_recall"] = test_metrics["macro_recall"]
    best_ckpt["test_macro_f1"] = test_metrics["macro_f1"]
    torch.save(best_ckpt, args.save_path)

    # Bramki oglądamy na teście, już po decyzji o wagach.
    # Rysunek niczego nie wybiera, więc nie zużywa testu jako walidacji.
    analyze_highway_gates(model, test_loader, device, save_path="highway_gates_analysis.png")

    print("\n--- Final Single Test Set Evaluation (Selected Best Val Epoch: " f"{best_epoch}) ---")
    print(f"Test Loss:            {test_metrics['loss']:.4f}")
    print(f"Test Accuracy:        {test_metrics['acc']:.2f}% (95% CI [{ci_lo:.2f}%, {ci_hi:.2f}%])")
    print(f"Test Macro Precision: {test_metrics['macro_precision']:.2f}%")
    print(f"Test Macro Recall:    {test_metrics['macro_recall']:.2f}%")
    print(f"Test Macro F1:        {test_metrics['macro_f1']:.2f}%")
    print("Confusion Matrix (10x10):")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
