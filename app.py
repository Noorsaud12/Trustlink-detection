import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# --- Load dataset ---
df = pd.read_csv("fake_social_media.csv")

# --- Add fake usernames ---
df["username"] = [f"user_{i}" for i in range(len(df))]

# --- Encode text columns ---
text_columns = df.select_dtypes(include=["object"]).columns
for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# --- Prepare features and target ---
X = df.drop(columns=["is_fake"])
y = df["is_fake"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Streamlit Interface ---
st.title("Fake Account Detection")

username = st.text_input("Enter username (e.g., user_10):")

if st.button("Check"):
    # البحث عن الحساب
    if username.startswith("user_"):
        try:
            user_index = int(username.replace("user_", ""))
        except:
            user_index = None
    else:
        user_index = None

    if user_index is None or user_index >= len(df):
        st.write("❌ Username not found in dataset")
    else:
        user_data = df.iloc[[user_index]]
        X_user = user_data.drop(columns=["is_fake"])
        prediction = model.predict(X_user)[0]

        # عرض التفاصيل
        st.subheader("📊 Account Details")
        st.write(user_data.drop(columns=["is_fake"]).T)

        # عرض النتيجة مع الأسباب
        if prediction == 1:
            st.warning("⚠️ Fake account detected")
            st.write("### Possible reasons:")
            if user_data["follow_unfollow_rate"].values[0] > 300:
                st.write("- High follow/unfollow rate")
            if user_data["suspicious_links_in_bio"].values[0] == 1:
                st.write("- Suspicious links in bio")
            if user_data["spam_comments_rate"].values[0] > 100:
                st.write("- Many spam comments")
            if user_data["posts_per_day"].values[0] > 5 or user_data["posts_per_day"].values[0] < 0.05:
                st.write("- Abnormal posting activity")
        else:
            st.success("✅ Real account detected")
            st.write("### Positive indicators:")
            st.write("- Balanced followers/following ratio")
            st.write("- Normal posting activity")
            st.write("- No suspicious links or spam comments")
from sklearn.utils import resample

# فصل المزيف والحقيقي من الداتا الأصلية df
fake_accounts = df[df["is_fake"] == 1]
real_accounts = df[df["is_fake"] == 0]

# أخذ 200 من كل نوع (مع إعادة أخذ عينات إذا ناقص)
fake_sample = resample(fake_accounts, replace=True, n_samples=200, random_state=42)
real_sample = resample(real_accounts, replace=True, n_samples=200, random_state=42)

# دمجهم مع بعض
balanced_df = pd.concat([fake_sample, real_sample])

# استخدمي balanced_df بدل df في التدريب
X = balanced_df.drop("is_fake", axis=1)
y = balanced_df["is_fake"]

# تقسيم البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج على البيانات المتوازنة
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
