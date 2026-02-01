from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")