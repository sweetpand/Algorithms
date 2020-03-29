def sentiment(self, tweet):
		"""
		This function calculates sentiment
		from our base on our cleaned tweets.
		Uses textblob to calculate polarity.
		Parameters:
		----------------
		arg1: takes in a tweet (row of dataframe)	
		----------------
		Returns: 
			Sentiment:
			1 is Positive
			0 is Neutral
		       -1 is Negative
		"""

		analysis = TextBlob(tweet)
		if analysis.sentiment.polarity > 0:
			return 1
		elif analysis.sentiment.polarity == 0:
			return 0
		else:
			return -1




	def save_to_csv(self, df):
		"""
		Save cleaned data to a csv for further
		analysis.
		Parameters:
		----------------
		arg1: Pandas dataframe
		"""
		try:
			df.to_csv("clean_tweets.csv")
			print("\n")
			print("csv successfully saved. \n")

		
		except Error as e:
			print(e)
		



	def word_cloud(self, df):
		"""
		Takes in dataframe and plots a wordclous using matplotlib
		"""
		plt.subplots(figsize = (12,10))
		wordcloud = WordCloud(
								background_color = 'white',
								width = 1000,
								height = 800).generate(" ".join(df['clean_tweets']))
		plt.imshow(wordcloud)
		plt.axis('off')
		plt.show()
