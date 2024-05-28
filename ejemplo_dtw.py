import matplotlib.pyplot as plt
import numpy as np

# Definir las funciones
def f(x):
    return  1-((90-x)/(90))**4

def g(x):
    return 1-((90+x)/(90))**4

# Crear un rango de valores para x en cada dominio específico
x1 = np.linspace(0, 90, 400)
x2 = np.linspace(-90, 0, 400)

# Calcular los valores de las funciones
y1 = f(x1)
y2 = g(x2)

# Crear el gráfico
plt.figure(figsize=(10, 6))

# Graficar la primera función en su dominio específico
plt.plot(x1, y1, label=r'$\theta_1$', color='blue')

# Graficar la segunda función en su dominio específico
plt.plot(x2, y2, label=r'$\theta_2$', color='red')

# Añadir título y etiquetas
#plt.title('Gráfico de las funciones en dominios específicos')
plt.xlabel('Ángulo de desvío de la cámara [°]', fontsize=18,fontname='Times New Roman')
plt.ylabel(r'$\theta$' , rotation=0, labelpad=20,fontsize=18,fontname='Times New Roman')

ticks = np.arange(-90, 91, 15)
tick_labels = [f'{int(tick)}°' for tick in ticks]
plt.xticks(ticks, tick_labels, fontsize=18, fontname='Times New Roman')

ticks_y = np.arange(0, 1.1, 0.2)
tick_labels_y = [f'{tick:.1f}' for tick in ticks_y]
plt.yticks(ticks_y, tick_labels_y, fontsize=18, fontname='Times New Roman')

# Añadir una leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()
