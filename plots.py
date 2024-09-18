import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
def sentiment_distribution(df):
    brands = df['Brand'].unique()
    colors = sns.color_palette("husl", len(brands))
    brand_palette = dict(zip(brands, colors))

    # Define the color palette for sentiments
    sentiment_palette = {'Positive': '#01bb39', 'Negative': '#f9766d', 'Neutral': '#619dff'}

    # Create the count plot with brands on the x-axis and sentiments differentiated by color
    sns.countplot(x='Brand', hue='Sentiment label', data=df, palette=sentiment_palette)

    # Add title and labels
    plt.xlabel("Brand")
    plt.ylabel("Count")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.02))

    # Improve x-axis labels if needed
    plt.xticks(rotation=45)  # Rotate labels if needed

    # Show the plot
    plt.show()
def sentiment_distribution_by_brand(df):
    sns.countplot(x='Sentiment label', hue='Model', data=df)
    plt.legend(loc='upper right', bbox_to_anchor=(1.50, 1.5))  # Example to move legend outside the plot
    plt.title("Sentiment distribution by Brand")
    plt.show()

def wordcloud(df):
    positive_reviews = df[df['Sentiment label'] == 'Positive']['Translated Review Text'].tolist()
    negative_reviews = df[df['Sentiment label'] == 'Negative']['Translated Review Text'].tolist()
    # Combine the reviews into a single string
    positive_text = ' '.join(positive_reviews)
    negative_text = ' '.join(negative_reviews)

    # Set up stopwords (to remove common words like 'the', 'is', etc.)
    stop_words = set(stopwords.words('english'))

    # Generate Word Clouds
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words,
                                   colormap='Blues').generate(positive_text)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words,
                                   colormap='Reds').generate(negative_text)

    # Plot Positive Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Reviews Word Cloud', fontsize=20)
    plt.show()

    # Plot Negative Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Reviews Word Cloud', fontsize=20)
    plt.show()

def word_frequency(df):

    # Words of interest
    words_of_interest = ['range', 'battery', 'software', 'cost', 'price', 'charging']

    # Function to tokenize and clean the text (removing stopwords and non-alphabetic tokens)
    def tokenize_and_clean(text):
        tokens = word_tokenize(text.lower())  # Tokenize and convert to lower case
        stop_words = set(stopwords.words('english'))
        cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return cleaned_tokens

    # Initialize dictionaries to store the results for each brand
    positive_results = {}
    negative_results = {}

    # Group by brand
    for brand, group in df.groupby('Brand'):
        # Tokenize and clean the reviews for each brand
        positive_reviews = df[(df['Sentiment label'] == 'Positive') & (df['Brand'] == brand)][
            'Translated Review Text'].tolist()
        negative_reviews = df[(df['Sentiment label'] == 'Negative') & (df['Brand'] == brand)][
            'Translated Review Text'].tolist()
        positive_text = ' '.join(positive_reviews)  # Join all positive reviews for the brand
        negative_text = ' '.join(negative_reviews)  # Join all negative reviews for the brand

        # Tokenize and clean the positive and negative reviews
        positive_tokens = tokenize_and_clean(positive_text)
        negative_tokens = tokenize_and_clean(negative_text)

        # Count occurrences of each word of interest in positive and negative tokens
        positive_freq = Counter(positive_tokens)
        negative_freq = Counter(negative_tokens)
        # Extract frequencies for the specified words
        positive_word_freq = {word: positive_freq.get(word, 0) for word in words_of_interest}
        negative_word_freq = {word: negative_freq.get(word, 0) for word in words_of_interest}
        # Merge 'price' and 'cost'
        positive_word_freq['Price'] = positive_word_freq.get('price', 0) + positive_word_freq.get('cost', 0)
        negative_word_freq['Price'] = negative_word_freq.get('price', 0) + negative_word_freq.get('cost', 0)

        # Remove original 'price' and 'cost' from the dictionaries
        del positive_word_freq['price']
        del positive_word_freq['cost']
        del negative_word_freq['price']
        del negative_word_freq['cost']

        # Convert to DataFrame for better visualization
        positive_freq_df = pd.DataFrame(list(positive_word_freq.items()), columns=['Word', 'Frequency']).sort_values(
            by='Frequency', ascending=False)
        negative_freq_df = pd.DataFrame(list(negative_word_freq.items()), columns=['Word', 'Frequency']).sort_values(
            by='Frequency', ascending=False)

        # Store the DataFrames in dictionaries with the brand as the key
        positive_results[brand] = positive_freq_df
        negative_results[brand] = negative_freq_df

    for brand in positive_results:
        print(f"Frequencies of Words in Positive Reviews for {brand}:\n", positive_results[brand])
        print(f"\nFrequencies of Words in Negative Reviews for {brand}:\n", negative_results[brand])
        print("\n" + "=" * 50 + "\n")


if __name__ == '__main__':
    df = pd.read_csv("EV_reviews.csv", on_bad_lines='skip', engine='python')
    sentiment_distribution(df)
    sentiment_distribution_by_brand(df)
    word_frequency(df)
    wordcloud(df)
