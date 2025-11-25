from .preprocess import build_clean_dataset, save_clean_dataset
def main():
    df_clean = build_clean_dataset()
    save_clean_dataset(df_clean)
    print(f"Saved cleaned dataset with {len(df_clean)} rows.")
if __name__ == "__main__":
    main()