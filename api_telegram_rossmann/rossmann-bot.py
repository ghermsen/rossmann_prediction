import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import requests
import seaborn as sns
import telegram

from io import BytesIO
from flask import Flask, request, Response

# constants

TOKEN = 'YOUR TOKEN HERE'
bot = telegram.Bot(token=TOKEN)

def send_message(chat_id, text):

	url = "https://api.telegram.org/bot{}/".format(TOKEN)
	url = url + "sendMessage?chat_id={}".format(chat_id)

	r = requests.post(url, json = {"text": text})
	print("Status Code {}".format(r.status_code))

	return None


def load_dataset(store_id):
	# loading test dataset
	df10 = pd.read_csv("test.csv")
	df_store_raw = pd.read_csv("store.csv")

	# merge test and store dataset
	df_test = pd.merge(df10, df_store_raw, how = "left", on = "Store")

	# choose store for prediction
	df_test = df_test[df_test["Store"] == store_id]

	if not df_test.empty:

		# remove closed days
		df_test = df_test[df_test["Open"] != 0]
		df_test = df_test[~df_test["Open"].isnull()]
		df_test = df_test.drop("Id", axis = 1)

		# convert Dataframe to json
		data = json.dumps(df_test.to_dict(orient = "records"))

	else:
		data = "error"

	return data

def predict(data):

	# api call

	# heroku request

	url = "https://gh-rossmann-prediction.herokuapp.com/rossmann/predict"
	header = {"Content-type": "application/json"}
	data = data

	r = requests.post(url, data = data, headers = header)
	print("Status Code {}".format(r.status_code))

	d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())

	return d1


def parse_message(message):
	chat_id = message["message"]["chat"]["id"]
	store_id = message["message"]["text"]

	store_id = store_id.replace("/","")

	try:
		store_id = int(store_id)

	except ValueError:
		store_id = "error"

	return chat_id, store_id


# initialize API
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

	if request.method == "POST":
		message = request.get_json()
		chat_id, store_id = parse_message(message)

		if store_id != "error":
			# loading data
			data = load_dataset(store_id)

			if data != "error":
				# prediction
				d1 = predict(data)

				# calculation
				d2 = d1[["store", "prediction"]].groupby("store").sum().reset_index()

				#send message
				intro = 'Sales prediction for store {} will be generated...'.format(d2['store'].values[0])
				send_message(chat_id, intro)

				# send lineplot
				fig = plt.figure()
				sns.lineplot(x = 'week_of_year', y = 'prediction', data = d1)
				plt.title('Weekly Sales Prediction for Store {}'.format(d2['store'].values[0]))
				plt.xlabel('Week Year')
				plt.ylabel('Sales Prediction (€)')
				buffer = BytesIO()
				fig.savefig(buffer, format='png')
				buffer.seek(0)
				bot.send_photo(chat_id=chat_id, photo=buffer)
				# send intro message
				msg = 'Store {} will sell {:,.2f} € for the next six weeks. To generate sales prediction for another store, please insert a new store ID.'.format(d2['store'].values[0],d2['prediction'].values[0])
				send_message(chat_id, msg)

				return Response('Ok', status = 200)

			else:
				send_message(chat_id, "Sorry, predictions for this store ID is not available.")
				return Response("Ok", status = 200)

		else:
			send_message(chat_id, "Hello! Please, insert a store ID (a number between 1 and 1115) to generate the sales prediction for the next six weeks. Any other message will return this message. Thank you!")
			return Response("Ok", status = 200)

	else:
		return "<h1> Gabriel Hermsen Rossmann Telegram BOT </h1>"

if __name__ == "__main__":
	port = os.environ.get("PORT", 5000)
	app.run(host = "0.0.0.0", port = port)
