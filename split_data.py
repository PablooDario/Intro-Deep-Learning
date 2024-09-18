from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import joblib
import os

# German Credit Data
def load_german_credit_score():
    # Fetch German Credit Dataset 
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    
    # data (as pandas dataframes) 
    X = statlog_german_credit_data.data.features 
    y = statlog_german_credit_data.data.targets 

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    german_credit_dataset = { 'X_train' : X_train, 'X_val' : X_val,
                            'y_train' : y_train, 'y_val' : y_val }
    
    # Crear directorio si no existe
    if not os.path.exists('data'):
        os.makedirs('data')

    # Guardar el dataset en un archivo
    joblib.dump(german_credit_dataset, 'data/german_credit_dataset.pkl')

    print('-'*20 + 'German Credit Dataset' + '-'*20)
    print(f"X_train_shape {X_train.shape}, y_train.shape {y_train.shape}")
    print(f"X_val_shape {X_val.shape}, y_val.shape {y_val.shape}")


# MNIST
def load_mnist_data():
    # Fetch MNIST Dataset
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mnist_dataset = { 'X_train' : X_train, 'X_val' : X_val,
                            'y_train' : y_train, 'y_val' : y_val }

    # Crear directorio si no existe
    if not os.path.exists('data'):
        os.makedirs('data')

    # Guardar el dataset en un archivo
    joblib.dump(mnist_dataset, 'data/mnist_dataset.pkl')

    print('-'*20 + 'MNIST Dataset' + '-'*20)
    print(f"X_train_shape {X_train.shape}, y_train.shape {y_train.shape}")
    print(f"X_val_shape {X_val.shape}, y_val.shape {y_val.shape}")


def main():
    load_german_credit_score()
    load_mnist_data()
    
if __name__ == '__main__':
    main()