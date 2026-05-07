import pandas as pd

# 1. Load the separate files
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# 2. Add a 'label' column so the AI knows which is which
# We will use 'REAL' for true news and 'FAKE' for fake news
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# 3. Combine (concatenate) them into one table
combined_df = pd.concat([true_df, fake_df], axis=0)

# 4. Shuffle the data (important for training)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# 5. Save it as news.csv for your detector
combined_df.to_csv('news.csv', index=False)

print("Successfully merged! Your 'news.csv' is ready.")