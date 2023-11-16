import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# from sklearn.preprocessing import OneHotEncoder

#Esta parte es para aplicar one-hot encoding a la columna de "species"

# data = pd.read_csv('Fish.csv')
# column_to_encode = 'Species'
# encoder = OneHotEncoder(sparse=False)
# encoded_data = encoder.fit_transform(data[[column_to_encode]])

# encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column_to_encode]))

# data = pd.concat([data, encoded_df], axis=1)
# data = data.drop(column_to_encode, axis=1)
# data.to_csv('Fish_New.csv', index=False)

# Carga el archivo CSV en un DataFrame de pandas
data = pd.read_csv('Fish.csv')

# Extrae las variables independientes y dependientes
X1 = data['Length2'].values
X2 = data['Height'].values
Y = data['Weight'].values

# Crea una matriz de diseño con las características y una columna de unos para el término constante
X_design = np.column_stack((np.ones(len(X1)), X1, X2, X1**2, X2**2, X1*X2))

X_design3 = np.column_stack((np.ones(len(X1)), X1, X2, X1**2, X2**2, X1*X2, X1**3, X2**3, (X1**2)*X2, X1*(X2**2)))

X_design4 = np.column_stack((np.ones(len(X1)), X1, X2, X1**2, X2**2, X1*X2, X1**3, X2**3, (X1**2)*X2, X1*(X2**2), X1**4, X2**4, (X1**3)*X2, (X1**2)*(X2**2), X1*(X2**3)))

# Calcula los coeficientes de la regresión usando la ecuación normal
coefficients = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y

coefficients3 = np.linalg.inv(X_design3.T @ X_design3) @ X_design3.T @ Y

coefficients4 = np.linalg.inv(X_design4.T @ X_design4) @ X_design4.T @ Y

# Imprime los coeficientes
print("Coeficientes de la regresión:")
print("Intercepto:", coefficients[0])
print("Coeficiente X1:", coefficients[1])
print("Coeficiente X2:", coefficients[2])
print("Coeficiente X1^2:", coefficients[3])
print("Coeficiente X2^2:", coefficients[4])
print("Coeficiente X1*X2:", coefficients[5])

# Predice los valores de Y usando los coeficientes
Y_pred = X_design @ coefficients

Y_pred3 = X_design3 @ coefficients3

Y_pred4 = X_design4 @ coefficients4

mse2 = np.mean((Y - Y_pred)**2)
mse3 = np.mean((Y - Y_pred3)**2)
mse4 = np.mean((Y - Y_pred4)**2)

print("Error Cuadrático Medio (MSE) 2do grado:", mse2)
print("Error Cuadrático Medio (MSE) 3er grado:", mse3)
print("Error Cuadrático Medio (MSE) 4to grado:", mse4)

# Grafica los resultados
plt.scatter(Y, Y_pred)
plt.xlabel("Valores reales de Y")
plt.ylabel("Valores predichos de Y")
plt.title("Regresión Polinomial Múltiple")
plt.show()
