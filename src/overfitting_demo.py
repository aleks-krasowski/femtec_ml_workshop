import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def which_model_is_better(show_test_data=False):

    fig, ax = plt.subplots(ncols = 2, figsize=(16,9))

    # Generating synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y = x + np.random.normal(0, 0.5, x.shape[0])

    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Polynomial Regression Model
    degree = 8  # Very high degree for overfitting
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(x_train[:, np.newaxis], y_train)



    # Visualizing the model performance
    ax[0].scatter(x_train, y_train, color='blue', label='Trainingsdaten')
    ax[0].plot(np.linspace(0, 10, 100), polyreg.predict(np.linspace(0, 10, 100)[:, np.newaxis]), color='red', label=fr'$f(x)$')
    
    ax[0].legend()
    ax[0].set_xlabel('y')
    ax[0].set_ylabel('x')
    ax[0].grid()

    ax[1].scatter(x_train, y_train, color='blue', label='Training data')
    ax[1].plot((0,10), (0, 10), color = 'red', label = '$f(x)$')
    ax[1].legend()
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('x')
    ax[1].grid()

    if show_test_data:
        ax[0].scatter(x_test, y_test, color='green', label='Testdaten')
        ax[1].scatter(x_test, y_test, color='green', label='Testdaten')
        y_pred = polyreg.predict(x_test[:, np.newaxis])
        mae_overfitted = mean_absolute_error(y_test, y_pred)
        mae_gt = mean_absolute_error(y_test, x_test)
        ax[0].set_title(f"Durschnitlicher Absoluter Fehler: {mae_overfitted:.3f} (AUF TESTDATEN)")
        ax[1].set_title(f"Durschnitlicher Absoluter Fehler: {mae_gt:.3f} (AUF TESTDATEN)")
    else:
        y_pred = polyreg.predict(x_train[:, np.newaxis])
        mae_overfitted = mean_absolute_error(y_train, y_pred)
        mae_gt = mean_absolute_error(y_train, x_train)
        ax[0].set_title(f"Durschnitlicher Absoluter Fehler: {mae_overfitted:.3f}")
        ax[1].set_title(f"Durschnitlicher Absoluter Fehler: {mae_gt:.3f}")