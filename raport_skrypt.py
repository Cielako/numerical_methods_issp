import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

#zdefiniowanie funkcji na podstawie wyprowadzonych równań
def f(t, y, c):
    return [y[1], -c*y[1] - g]

#funkcja w przypadku gdy nie uwzględniamy oporu powietrza
def f2(t, y):
    return [y[1], -9.81]

#analityczne rozwiązanie problemu do porównania wysokości z naszym numerycznym rozwiązaniem
def a_yt(t):
    return ((((m*v_0/k) + ((m**2)*g/k**2)) * (1 - np.exp(-k*t/m))) - m*g*t/k)+100

#analityczne rozwiązanie problemu do porównania prędkości z naszym numerycznym rozwiązaniem
def a_vt(t):
    return  (np.e**(-k*t/m)*(k*v_0 - g*m*(np.e**(k*t/m)-1)))/k

m = 50          #masa pocisku (w kilogramach)
k = 5           #pierwotny współczynnik oporu (użyty w funckji analitycznej)
C = k/m         #wspolczynnik oporu po przekształceniach
y_0 = 100       #początkowa wysokość (w metrach)
v_0 = 60        #przyjęta prędkość początkowa
t_max = 13      #maksymalny czas
g = 9.81        #przyspieszenie grawitacyjne

y0 = [y_0, v_0] #warunki początkowe

t_range = np.arange(0, t_max, 0.001) #zakres czasu użyty w sprawdzeniu analitycznym i w rzucie bez oporu

#zastosowanie funkcji solve_ivp() do rozwiązania naszych równań różniczkowych
w = solve_ivp(lambda t,y:f(t,y,C), [0, t_max], y0, atol=1e-12, rtol=1e-13)

#wyznaczenie rozwiązań dla przypadku gdy nie mamy oporu powietrza
w_op = solve_ivp(lambda t,y:f2(t,y), [0, t_max], y0, t_eval=t_range)


#uzyskanie maksymalnej wysokości osiągniętej przez pocisk to po prostu wyciągnięcie maksymalnej wartości w y[0]
maks_h = max(w.y[0]) #wysokość maksymalna uzyskana numerycznie
maks_ha = max(a_yt(t_range)) #wysokość maksymalna uzyskana analitycznie

print(f'Maksymalna wysokość uzyskana numerycznie: {maks_h}')
print(f'Maksymalna wysokość uzyskana analitycznie: {maks_ha}')
print(f'Błąd: {maks_ha - maks_h}')

for i in range(75):
     print("%4d  wysokość: %18.15f" %(i, w.y[0][i]))  


#pobieram z naszego numerycznego wyniku kilka najbliższych punktów
#do poziomu tarasu aby później wyznaczyć czas upadku
taras_y = []
taras_t = []
for i in range(len(w.y[0])):
    if w.y[0][i] < 110 and w.y[0][i] > 98 and w.t[i] > 10 and w.t[i] < 12:
        taras_y.append(w.y[0][i])
        taras_t.append(w.t[i])

#używam interpolacji wielomianowej lagrange
wiel = lagrange(taras_t, taras_y)

#z utworzonego wielomianu wyciągam współczynniki
wsp = Polynomial(wiel).coef

#w ostatnim współczynniku odejmuje 100 ponieważ nasz 'pierwiastek' znajduje się na wysokości 100m
wsp[-1] = wsp[-1] - 100

#za pomocą roots obliczam pierwiastki
punkty = np.roots(wsp)

#w tym warunku sprawdzam który wyliczony czas upadku jest zbliżony do prawdopodobnego czasu upadku
t_upadku = 0
for i in range(len(punkty)):
    if punkty[i] < 12 and punkty[i] > 10:
        t_upadku = punkty[i]
#w ten sposób uzyskaliśmy czas upadku, który posłuży nam do sprawdzenia prędkości w momencie upadku
print(f'Czas, w momencie którego pocisk upadł na taras: {t_upadku} s')

#tak samo jak w przypadku wyznaczania czasu upadku, tutaj bierzemy kilka najbliższych wartości prędkości,
#w których przypuszczając spadł pocisk, przypuszczenie przedziału dokonane podstawie wykresu i wyznaczonego czasu upadku
v_upadek = []
t_upadek_v = []
for i in range(len(w.t)):
    if w.y[1][i] > -42.9 and w.y[1][i] < -41.5:
        v_upadek.append(w.y[1][i])
        t_upadek_v.append(w.t[i])

#używam interpolacji wielomianowej lagrange
wiel_v = lagrange(t_upadek_v, v_upadek)

#dzięki temu jesteśmy w stanie podstawić nasz czas upadku do uzyskanego wielomianu i uzyskać prędkość upadku
v_upadku = wiel_v(t_upadku)

print(f'Prędkość upadku uzyskana numerycznie: {v_upadku} m/s')
print(f'Prędkość upadku uzyskana analitycznie: {a_vt(t_upadku)} m/s')
print(f'Błąd: {v_upadku - a_vt(t_upadku)}')

for i in range(75):
     print("%4d  prędkość: %18.15f" %(i, w.y[1][i]))  

#rysowanie wykresów
plot1 = plt.figure(1)
plt.plot(t_range, a_yt(t_range), color='y', linewidth=5, zorder=1)
plt.scatter(w.t, w.y[0], color='b', s=3, zorder=2)
plt.xlabel('czas [s]')
plt.ylabel('wysokość [m]')
plt.axhline(y=100, color='r')
plt.grid()
plt.title('Rzut pionowy z oporem powietrza')
plt.legend(['analitycznie','poziom tarasu', 'numerycznie'], loc = 'best')

plot2 = plt.figure(2)
plt.plot(t_range, a_vt(t_range), color='y', linewidth=5, zorder=1)
plt.scatter(w.t, w.y[1], color='b', s=2, zorder=2)
plt.axhline(y=0, color='r')
plt.title('Prędkość w rzucie pionowym z oporem powietrza')
plt.legend(['analitycznie', 'moment spadania', 'numerycznie'], loc = 'best')
plt.xlabel('czas [s]')
plt.ylabel('prędkość [m/s]')
plt.grid()

plot3 = plt.figure(3)
plt.plot(w_op.t, w_op.y[0], color='g')
plt.scatter(w.t, w.y[0], color='b', s=2)
plt.axhline(y=100, color='r')
plt.title('Rzut pionowy bez i z oporem powietrza')
plt.legend(['bez oporu', 'poziom tarasu', 'z oporem'], loc = 'best')
plt.xlabel('czas [s]')
plt.ylabel('wysokość [m]')
plt.grid()

plt.show()

