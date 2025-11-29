import pandas as pd
import numpy as np
from io import StringIO

# Your data with correct column names
filepath ='dataset/step3_data_majority_features.csv'


df = pd.read_csv(filepath)

# Handle missing Screens value
df['Screens'] = df['Screens'].fillna(df['Screens'].median())

# CREATE RATINGS LABEL FIRST (This is what you're trying to predict)
# Convert continuous Ratings to categorical
# df['Ratings_Label'] = pd.cut(df['Ratings'], 
#                               bins=[0, 5.5, 6.5, 10], 
#                               labels=['Poor', 'Average', 'Good'])
label_encoding = {'Poor': 0, 'Average': 1, 'Good': 2}
df['Rating_Label_Encoded'] = df['Rating_Label'].map(label_encoding)

print("=" * 100)
print("COMPLETE CORRELATION ANALYSIS WITH CORRECT COLUMN NAMES")
print("=" * 100)

print("\n📊 DATASET OVERVIEW")
print("-" * 100)
print(f"Total movies: {len(df)}")
print(f"Features: {', '.join(df.columns[1:])}")  # Skip Ratings
print(f"\nTarget variable: Rating_Label")
print(f"Target distribution:\n{df['Rating_Label'].value_counts()}")

print("\n📈 CORRELATION WITH RATINGS (Sorted by Strength)")
print("-" * 100)

# Select numeric columns for correlation
numeric_features = ['Budget','Aggregate Followers','Comments','Dislikes','Screens','Sentiment',]


# Correlate with NUMERIC Ratings (not categorical label)
correlations = df[numeric_features].corrwith(df['Rating_Label_Encoded']).sort_values(ascending=False)

print(f"\n{'Feature':<25} {'Correlation':>12} {'Strength':>20} {'Direction'}")
print("-" * 100)
for feature, corr in correlations.items():
    abs_corr = abs(corr)
    if abs_corr > 0.7:
        strength = "🔥 STRONG"
    elif abs_corr > 0.5:
        strength = "⭐ MODERATE"
    elif abs_corr > 0.3:
        strength = "💫 WEAK"
    elif abs_corr > 0.15:
        strength = "💤 VERY WEAK"
    else:
        strength = "❌ NEGLIGIBLE"
    
    direction = "↑ Positive" if corr > 0 else "↓ Negative"
    print(f"{feature:<25} {corr:>12.4f} {strength:>20} {direction}")

print("\n🔧 FEATURE ENGINEERING & DERIVED FEATURES")
print("-" * 100)

# # Create engineered features (handle division by zero)
# df['ROI'] = ((df['Gross'] - df['Budget']) / df['Budget'].replace(0, 1)) * 100
# df['Profit'] = df['Gross'] - df['Budget']
# df['Like_Ratio'] = df['Likes'] / (df['Likes'] + df['Dislikes']).replace(0, 1)
# df['Engagement_Rate'] = (df['Comments'] + df['Likes'] + df['Dislikes']) / df['Views'].replace(0, 1)
# df['View_per_Screen'] = df['Views'] / df['Screens'].replace(0, 1)
# df['Gross_per_Screen'] = df['Gross'] / df['Screens'].replace(0, 1)
# df['Comment_Rate'] = df['Comments'] / df['Views'].replace(0, 1)
# df['Like_per_View'] = df['Likes'] / df['Views'].replace(0, 1)
# df['Dislike_Rate'] = df['Dislikes'] / df['Views'].replace(0, 1)

# engineered = ['ROI', 'Profit', 'Like_Ratio', 'Engagement_Rate', 'View_per_Screen', 
#               'Gross_per_Screen', 'Comment_Rate', 'Like_per_View', 'Dislike_Rate']

# print(f"\n{'Engineered Feature':<25} {'Correlation':>12} {'Strength'}")
# print("-" * 100)
# for feature in engineered:
#     corr = df[feature].corr(df['Dislikes'])  # Use numeric Ratings
#     abs_corr = abs(corr)
#     if abs_corr > 0.7:
#         strength = "🔥 STRONG"
#     elif abs_corr > 0.5:
#         strength = "⭐ MODERATE"
#     elif abs_corr > 0.3:
#         strength = "💫 WEAK"
#     else:
#         strength = "💤 VERY WEAK"
#     print(f"{feature:<25} {corr:>12.4f} {strength}")

# print("\n🎯 TOP PREDICTIVE FEATURES")
# print("-" * 100)

# # Combine all features
all_features = numeric_features 


all_correlations = df[all_features].corrwith(df['Rating_Label_Encoded']).abs().sort_values(ascending=False)

print("\nTop 10 features by correlation strength:")
for i, (feature, corr) in enumerate(all_correlations.head(10).items(), 1):
    print(f"{i:2d}. {feature:<25} {corr:.4f}")

print("\n🎓 MODEL RECOMMENDATIONS")
print("=" * 100)

max_correlation = all_correlations.max()
print(f"\nStrongest correlation: {max_correlation:.4f}")

if max_correlation > 0.5:
    print("✅ GOOD PREDICTABILITY - Proceed with modeling")
    print("   Recommended: Decision Tree, SVM, Random Forest")
else:
    print("⚠️  MODERATE PREDICTABILITY")
    print("   Use ensemble methods and feature engineering")