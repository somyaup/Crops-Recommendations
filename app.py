from flask import Flask
from flask import render_template,redirect,url_for,send_file
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import io
import random
from flask import Response,Flask
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



app = Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    #return render_template("index.html")
    return "HOME"

@app.route('/train', methods=['GET','POST'])
def log():
    df = pd.read_csv('data.csv')
    label = preprocessing.LabelEncoder()
    uni_label=df['label'].unique()
    print(uni_label)
    df['label'] = label.fit_transform(df['label'])
    print(df['label'].unique())
    df.head()
    df = df.iloc[:, 1:]
    df_norm = (df - df.mean()) / df.std()
    X = df_norm.iloc[:, :6]
    Y = df_norm.iloc[:, -1]
    model = get_model()
    model.summary()
    fig = plot_loss(X,Y)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
#-------------------------------------------------------
def convert_label_value(pred):
    df = pd.read_csv('data.csv')
    label = preprocessing.LabelEncoder()
    df['label'] = label.fit_transform(df['label'])
    y_mean = df["label"].mean()
    y_std = df["label"].std()
    return int(pred * y_std + y_mean)
#-------------------------------------------------------
def plot_loss(X,Y):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    X_train, X_test, y_train, y_test = split_data( X, Y, 0.1)
    model = get_model()
    preds_on_untrained = model.predict(X_test)
    path = './Checkpoints/my_cp1'
    model.load_weights(path)
    preds_on_trained =model.predict(X_test)

    price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
    price_on_trained = [convert_label_value(y) for y in preds_on_trained]
    price_y_test = [convert_label_value(y) for y in y_test]
    xs = price_y_test
    ys = price_on_untrained
    xt = price_y_test
    yt = price_on_trained
    axis.scatter(xs, ys)
    axis.scatter(xt, yt)
    axis.legend(['untrained', 'trained'], loc='upper left')

    return fig
    plt.xlabel('Y test')
    plt.ylabel('prediction')
#---------------------------------------------------------------------
def split_data(X,Y,test_size=0.1):
    X_arr = X.values
    Y_arr = Y.values
    print('X_arr shape: ', X_arr.shape)
    print('Y_arr shape: ', Y_arr.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size=test_size, shuffle=True, random_state=0)
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('X_test shape: ', X_test.shape)
    print('y_test shape: ', y_test.shape)
    return  X_train, X_test, y_train, y_test
    # -------------------------------------------------------
def get_model():
    model = Sequential([
        Dense(128, input_shape=(6,), activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1)
    ])
    model.compile(
        loss='mse',
        optimizer='adadelta'
    )
    return model

#-----------------------------------------------------------
@app.route('/get_image', methods=['GET','POST'])
def get_image():
    return render_template("index.html")
    #return redirect(url_for('static', filename='plot1'), code=301)


app.run()
