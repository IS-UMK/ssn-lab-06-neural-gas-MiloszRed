[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/RXuu2PeB)
# SSN. Lab. 6. Kwantyzacja wektorowa

Zapoznaj się z zawartością notatnika Jupyter umieszczonego w repozytorium  i wykonaj zawarte w nim ćwiczenia.

Notatnik: [06_vq.ipynb](https://github.com/IS-UMK/ssn_lab_06/blob/master/06_vq.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IS-UMK/ssn_lab_06/blob/master/06_vq.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IS-UMK/ssn_lab_06/master?filepath=06_vq.ipynb)

---
## Zadanie 6: Gaz neuronowy

Zaimplementuj algorytm gazu neuronowego służącego do kwantyzacji wektorowej. W implementacji możesz wykorzystać klasę ``VectorQuantization`` z pliku [vq.py](vq.py) i oraz implementację kwantyzacji wektorowej z ćwiczeń, wówczas rozwiązanie sprowadza się do przeciążenia metody ``fit()``. 

**Parametry początkowe** (ustawiane w argumentach konstruktora):

* $N$ liczba epok
* $k$ liczba neuronów (prototypów)
* $\eta_0$ początkowa wartość kroku uczenia
* $\lambda_0$ początkowa wartość promienia sąsiedztwa
* $\eta_{min}$ końcowa wartość kroku uczenia
* $\lambda_{min}$ końcowa wartość promienia sąsiedztwa

**Algorytm gazu neuronowego**  

1. Zainicjuj wagi $\mathbf{w}_i$ neuronów  (pozycje prototypów)
2. Ustaw $t=0$ oraz $T$ równe całkowitej liczbie iteracji (iloczyn liczby epok $N$ i liczby przypadków uczących)
3. Powtarzaj $N$ razy:
4. <ul>Dla każdego przypadku $\mathbf{x}$ ze zbioru uczącego wykonaj</ul>
5. <ul><ul>wyznacz aktualny współczynnik uczenia <br> $\eta(t)=\eta_0\left(\frac{\eta_{\min }}{\eta_0}\right)^{\frac{t}{T}}$
    </ul></ul>
6. <ul><ul>wyznacz aktualny promień sąsiedztwa <br> $\lambda(t)=\lambda_0\left(\frac{\lambda_{\min }}{\lambda_0}\right)^{\frac{t}{T}}$
    </ul></ul>
7. <ul><ul>uporządkuj neurony względem odległości od $\mathbf{x}$ (<i> wskazówka: kolejność prototypów można uzyskać funkcją <a href="https://numpy.org/doc/stable/reference/generated/numpy.argsort.html">numpy.argsort</a> </i>)</ul></ul>  
8. <ul><ul>zaktualizuj wagi <br> 
   $\mathbf{w}_{i} \leftarrow \mathbf{w}_{i} + \eta(t) \cdot h_i(t) \cdot \left(\mathbf{x}-\mathbf{w}_i\right)\,$<br> gdzie $\, h_i(t)=e^{-\frac{m(i)}{\lambda(t)}}$, zaś 
   $\, m(i) = 0, 1, \ldots, k-1$ oznacza pozycję neuronu $\mathbf{w}_i$ w rankingu odległości od najbliższego do najdalszego
    </ul></ul>
1. <ul><ul>zwiększ $t = t + 1$</ul></ul>

Wykorzystaj algorytm gazu neuronowego do kompresji obrazu [dane/Lenna.png](dane/Lenna.png). Do przygotowania zbioru uczącego możesz wykorzystać funkcje ``img_to_vectors`` oraz ``vectors_to_image`` dostępne w pliku [utils.py](utils.py)  
Spróbuj dobrać parametry modelu gazu neuronowego tak aby uzyskać jak najmniejszy błąd rekonstrukcji przy jak najmniejszym rozmiarze księgi kodów (liczby prototypów). 

Rozwiązanie w postaci notatnika Jupyter (``.ipynb``) lub skryptu w języku Python (``.py``) w repozytorium GitHub.





