def train_model(dataloader, tokenizer):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(2):
        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"].squeeze(1).to(device)
            target_ids = batch["target_ids"].squeeze(1).to(device)

            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    return model
