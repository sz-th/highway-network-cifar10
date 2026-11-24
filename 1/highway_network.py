"""
Rodzina modeli, czyli zbiór funkcji z wagami.

Wykład 1 mówi, że sieć neuronowa to złożona rodzina funkcji z parametrami,
uczona przez schodzenie po nachyleniu błędu. Ten plik tę rodzinę zapisuje.
Nie uczy wag i nie liczy metryk. Tym zajmuje się train_cifar10.py.

Wykład 2 otwiera najprostszą taką funkcję: wynik liniowy z = w^T x + b,
potem nieliniowość. Tu jest tego dużo, ułożone jedno za drugim.
Warstwa Highway dokłada do tego bramkę. Bramka decyduje, ile wziąć
z nowego przekształcenia, a ile przepuścić wejście bez zmiany.
Dzięki ścieżce „bez zmiany” głęboki stos da się w ogóle uczyć:
gradient nie ginie po drodze, nawet gdy bramka jest prawie zamknięta.

Na samym końcu zostaje zwykła warstwa liniowa na 10 klas.
Wykład 2 nazywa to regresją softmax: zero ukrytych warstw na gotowym
wektorze cech, decyzja to klasa z największą liczbą.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayLinear(nn.Module):
    """
    Jedna warstwa Highway na zwykłym wektorze.

    Wejście i wyjście mają tyle samo liczb, bo na końcu dodajemy
    nowe cechy do oryginalnego x. Inaczej kształty by się nie zgadzały.

    Dwie osobne warstwy liniowe:
    H liczy nowe cechy, T liczy bramkę.
    Bramka przechodzi przez sigmoid z wykładu 2, więc każda jej
    liczba leży między 0 a 1.
    """

    def __init__(self, features, transform_bias=-2.0, activation=F.relu):
        super().__init__()
        # H: nowe cechy. Tyle wejść, ile wyjść, żeby dało się dodać do x.
        self.H = nn.Linear(features, features)
        # T: surowa bramka, zanim sigmoid ściśnie ją do (0, 1).
        self.T = nn.Linear(features, features)
        # Domyślnie ReLU. Obcina ujemne wartości nowej cechy do zera.
        self.activation = activation
        # Te dwa pola żyją tylko przy oglądaniu sieci, nie przy uczeniu.
        self.last_gate = None
        self.last_output = None
        self._init_weights(transform_bias)

    def _init_weights(self, transform_bias):
        # Skala startowa wag. Tego wzoru nie ma na wykładach 1–2.
        # Chodzi o to, co wykład 2 pokazał na Irisie: pierwsze liczby
        # w sieci muszą mieć rozsądną wielkość. Inaczej jeden krok
        # uczenia nie pasuje naraz do wszystkich warstw.
        nn.init.kaiming_normal_(self.H.weight, mode="fan_in", nonlinearity="relu")
        # Bias H startuje od zera. Na początku nie przesuwamy cech na siłę.
        nn.init.zeros_(self.H.bias)
        nn.init.kaiming_normal_(self.T.weight, mode="fan_in", nonlinearity="linear")
        # Ujemny bias bramki. Przy wagach T bliskich zera
        # bramka startuje koło sigmoid(-2) ≈ 0,12.
        # Około 88% sygnału idzie wtedy ścieżką „bez zmiany”.
        # Sieć na starcie nie jest stosem losowych przekształceń.
        # Pochodna sigmoidu, z wykładu 2: sigma(z) * (1 - sigma(z)).
        # Największa w zerze (1/4), mała w ogonach.
        # Przy -2 bramka jeszcze ma z czego się uczyć.
        # Przy -4 jest już prawie płaska i prawie zamknięta.
        nn.init.constant_(self.T.bias, transform_bias)

    def forward(self, x):
        # Nowe cechy: warstwa liniowa, potem ReLU.
        h = self.activation(self.H(x))
        # Bramka. Sigmoid z wykładu 2: sigma(z) = 1 / (1 + e^(-z)).
        # Wynik zawsze w (0, 1). sigma(-z) = 1 - sigma(z),
        # więc udział ścieżki „bez zmiany” to po prostu 1 - t.
        # Druga bramka nie potrzebuje własnych wag.
        t = torch.sigmoid(self.T(x))
        # Mieszanka, osobno na każdej współrzędnej.
        # t blisko 1: bierzemy h. t blisko 0: zwracamy x.
        # Pośrodku: kawałek jednego i drugiego.
        out = h * t + x * (1.0 - t)
        # Przy sprawdzaniu sieci zapamiętujemy bramkę i wyjście,
        # żeby później narysować, co warstwa naprawdę robi.
        # detach odcina tę kopię od uczenia. W treningu tego nie ruszamy,
        # bo zapis nie może zmieniać gradientu.
        if not self.training:
            self.last_gate = t.detach()
            self.last_output = out.detach()
        return out


class HighwayConv2d(nn.Module):
    """
    Ta sama mieszanka co HighwayLinear, tylko na obrazku.

    Wejście ma kształt (paczka, kanały, wysokość, szerokość).
    Splot 3x3 z ramką 1 piksel patrzy na sąsiadów i nie zmienia
    wysokości ani szerokości. Bez ramki h byłoby mniejsze niż x
    i nie dałoby się ich dodać.

    Splot to lokalna mapa liniowa ze wspólnymi wagami na całym obrazie.
    Pełne omówienie splotu jest dopiero na wykładzie 7.
    Tu wystarczy: te same wagi w każdym miejscu, kształt zostaje.
    """

    def __init__(self, channels, kernel_size=3, transform_bias=-2.0, use_bn=True):
        super().__init__()
        # kernel 3 daje padding 1. Mapa 32x32 zostaje 32x32.
        padding = kernel_size // 2
        # Gdy jest batch norm, bias splotu H jest zbędny:
        # normalizacja i tak odejmie średnią, a własny shift ma w środku.
        self.H = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)
        # Batch norm to standaryzacja w środku sieci, osobno na każdym kanale.
        # Ten sam pomysł co centrowanie cech na wykładzie 2:
        # bez podobnej skali jeden krok uczenia działa nierówno.
        # W treningu używa średniej z paczki. Przy sprawdzaniu
        # używa średniej zapamiętanej z treningu.
        # Dlatego ewaluacja woła model.eval().
        self.bn_h = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        # Bramka zostaje zwykłym splotem z biasem. Bias startuje ujemny.
        self.T = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.last_gate = None
        self.last_output = None
        self._init_weights(transform_bias)

    def _init_weights(self, transform_bias):
        # fan_out: skala wag liczona od liczby wyjść splotu.
        # Cel ten sam co w warstwie liniowej: pierwsza aktywacja
        # nie może wybuchnąć ani zgasnąć.
        nn.init.kaiming_normal_(self.H.weight, mode="fan_out", nonlinearity="relu")
        if self.H.bias is not None:
            nn.init.zeros_(self.H.bias)
        nn.init.kaiming_normal_(self.T.weight, mode="fan_out", nonlinearity="linear")
        # Ten sam start bramki co w HighwayLinear: koło sigmoid(-2).
        nn.init.constant_(self.T.bias, transform_bias)

    def forward(self, x):
        # Najpierw splot, potem normalizacja kanału, potem ReLU.
        h = F.relu(self.bn_h(self.H(x)))
        # Bramka ma kształt całego obrazka: inna wartość
        # w każdym kanale i w każdym pikselu.
        t = torch.sigmoid(self.T(x))
        out = h * t + x * (1.0 - t)
        if not self.training:
            self.last_gate = t.detach()
            self.last_output = out.detach()
        return out


class PlainTransitionConv2d(nn.Module):
    """
    Zwykły splot bez bramki. Tylko on zmienia liczbę kanałów.

    Warstwa Highway dodaje h do x, więc oba muszą mieć
    tyle samo kanałów. Przejście 3 -> 32, albo 32 -> 48,
    robi się tutaj: splot, normalizacja, ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvHighwayNet(nn.Module):
    """
    Sieć splotowa na CIFAR-10, złożona z kilku etapów.

    Każdy etap:
    1. zwykły splot zmienia liczbę kanałów,
    2. kilka warstw Highway zostawia tę liczbę w spokoju,
    3. między etapami obraz jest zmniejszany 2 razy (32 -> 16 -> 8).

    Na końcu mapa jest uśredniana do jednego wektora
    i warstwa liniowa daje 10 liczb, po jednej na klasę.
    To jest regresja softmax z wykładu 2.
    Granica klas w przestrzeni tego wektora jest płaska.
    Krzywą w przestrzeni pikseli robią wcześniejsze warstwy,
    bo wektor jest nieliniową funkcją obrazu.
    Jedna warstwa liniowa sama nie liczy XOR.
    Wykład 2 pokazuje to jako za małą rodzinę funkcji,
    czyli czysty bias z wykładu 1.
    Głębokość sieci siedzi przed klasyfikatorem, nie w nim.
    """

    def __init__(
        self,
        num_classes=10,
        stage_channels=(32, 64, 128),
        stage_highway_layers=(3, 4, 3),
        transform_bias=-2.0,
        dropout_p=0.2,
        use_bn=True,
    ):
        super().__init__()
        stages = []
        # CIFAR-10: trzy kanały koloru na wejściu.
        in_ch = 3
        for stage_idx, (out_ch, num_hw) in enumerate(zip(stage_channels, stage_highway_layers)):
            # Najpierw zmiana liczby kanałów, bez bramki.
            layers = [PlainTransitionConv2d(in_ch, out_ch, kernel_size=3, use_bn=use_bn)]
            # Potem stos Highway o stałej liczbie kanałów.
            for _ in range(num_hw):
                layers.append(HighwayConv2d(out_ch, kernel_size=3, transform_bias=transform_bias, use_bn=use_bn))
            # Ostatni etap nie zmniejsza obrazka. Zostaje go uśrednić.
            if stage_idx < len(stage_channels) - 1:
                # Okno 2x2, krok 2: wysokość i szerokość maleją dwa razy.
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                # Dropout jest w księdze z wykładu 1: narzędzie przeciw
                # zapamiętywaniu szumu. Kosztuje trochę dopasowania
                # do treningu, zmniejsza rozrzut modelu między zbiorami.
                # Wzór jest odłożony na wykład 6.
                # W treningu losowo zeruje część kanałów.
                # Przy sprawdzaniu przepuszcza pełną mapę.
                # Przełącza to model.train() / model.eval().
                if dropout_p > 0:
                    layers.append(nn.Dropout2d(p=dropout_p))
            stages.append(nn.Sequential(*layers))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)
        # Z mapy zostaje jedna liczba na kanał: średnia po pikselach.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 10 liczb, zwanych logitami. To jeszcze nie prawdopodobieństwa.
        # Softmax i strata dokładają się dopiero w pętli uczenia.
        self.classifier = nn.Linear(stage_channels[-1], num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        # Z (paczka, kanały, 1, 1) zostaje (paczka, kanały).
        x = torch.flatten(x, 1)
        # z = W h + b. Decyzja, już poza tym plikiem, to argmax.
        return self.classifier(x)

    def get_highway_layers(self):
        # Wszystkie warstwy Highway, po kolei od wejścia do wyjścia.
        # Rysunek bramek czyta z nich last_gate.
        return [m for m in self.modules() if isinstance(m, HighwayConv2d)]


class DeepFCHighwayNet(nn.Module):
    """
    Ta sama idea bez splotu: obraz spłaszczony do 3072 liczb
    (32 * 32 * 3), rzut na 50 jednostek, potem stos HighwayLinear.

    Służy do porównania z głęboką siecią bez bramek.
    Bramka zmienia krajobraz błędu, nie metrykę.
    """

    def __init__(self, input_dim=3072, hidden_dim=50, num_highway_layers=19, num_classes=10, transform_bias=-2.0):
        super().__init__()
        # Pierwsza warstwa może zmienić wymiar. Dalsze Highway już nie.
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.input_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.input_layer.bias)

        self.highway_layers = nn.ModuleList(
            [HighwayLinear(hidden_dim, transform_bias=transform_bias) for _ in range(num_highway_layers)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # Obraz (paczka, 3, 32, 32) -> wektor (paczka, 3072).
        x = torch.flatten(x, 1)
        x = F.relu(self.input_layer(x))
        for layer in self.highway_layers:
            x = layer(x)
        return self.classifier(x)

    def get_highway_layers(self):
        return list(self.highway_layers)


class DeepFCPlainNet(nn.Module):
    """
    Ten sam stos głębokości, bez bramki: warstwa liniowa i ReLU.

    71 jednostek, a nie 50, bo jedna zwykła warstwa 71x71
    ma prawie tyle wag, co jedna warstwa Highway 50x50
    z dwiema macierzami (H i T). Porównanie jest o bramce,
    nie o tym, że jedna sieć ma po prostu więcej liczb do ruszenia.

    Przy zamkniętej bramce Highway przepuszcza gradient w stosunku 1:1.
    Zwykły stos ReLU takiej ścieżki nie ma. Na dużej głębokości
    minimalizacja błędu potrafi w nim stanąć, choć rodzina funkcji
    na papierze jest wystarczająco duża.
    """

    def __init__(self, input_dim=3072, hidden_dim=71, num_hidden_layers=19, num_classes=10):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.input_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.input_layer.bias)

        layers = []
        for _ in range(num_hidden_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            nn.init.kaiming_normal_(lin.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.hidden_layers = nn.ModuleList(layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            # Każda warstwa obowiązkowo przekształca. Nie ma ścieżki x.
            x = F.relu(layer(x))
        return self.classifier(x)

    def get_highway_layers(self):
        # Bramek nie ma. Analiza bramek w train_cifar10.py wtedy nic nie rysuje.
        return []


def build_highway11_cifar10(transform_bias=-2.0, use_bn=True):
    """
    Mniejsza sieć na CIFAR-10.

    Kanały etapów: 32, 48, 64.
    Warstwy Highway w etapach: 2, 3, 2, razem 7.
    Nazwa „11” liczy warstwy z wagami:
    3 przejścia + 7 Highway + klasyfikator.
    Dropout 0,1 między etapami.
    """
    return ConvHighwayNet(
        num_classes=10,
        stage_channels=(32, 48, 64),
        stage_highway_layers=(2, 3, 2),
        transform_bias=transform_bias,
        dropout_p=0.1,
        use_bn=use_bn,
    )


def build_highway19_cifar10(transform_bias=-2.0, use_bn=True):
    """
    Większa rodzina funkcji. Wykład 1 nazywa to pokrętłem pojemności:
    większa rodzina łatwiej dopasuje trening, a łatwiej też zapamięta szum.
    Czy urosła za bardzo, widać po luce między krzywą treningu a walidacji.

    Kanały: 64, 96, 128. Po 5 warstw Highway w każdym etapie, razem 15.
    3 przejścia + 15 Highway + klasyfikator = 19.
    """
    return ConvHighwayNet(
        num_classes=10,
        stage_channels=(64, 96, 128),
        stage_highway_layers=(5, 5, 5),
        transform_bias=transform_bias,
        dropout_p=0.1,
        use_bn=use_bn,
    )


def count_parameters(model):
    # Liczba wag, które uczenie naprawdę rusza.
    # Wykład 1: theta to wszystkie parametry modelu.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
