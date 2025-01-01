if __name__ == "__main__":
    dataloader, tokenizer = load_and_prepare_data()
    model = train_model(dataloader, tokenizer)

    sample_text = """Chapter 1: The heart of a demon never has regret even in death "Fang Yuan, quietly hand over the Spring Autumn Cicada and I'll give you a quick death!" "Old bastard Fang, stop attempting to resist anymore, today all of the major factions of justice have combined together just to destroy your devil lair. This place is already covered in inescapable nets, this time you will definitely be decapitated!"""
    summary = generate_summary(model, tokenizer, sample_text)
    print("Generated Summary:", summary)
