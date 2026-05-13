import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.cnn_lstm.model import CNNLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="model.pt")
    parser.add_argument("--max-len", type=int, default=48)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    stoi = ckpt["vocab_stoi"]
    itos = {v: k for k, v in stoi.items()}

    pad_idx = stoi.get("<pad>", 0)
    bos_idx = stoi.get("<bos>", 1)
    eos_idx = stoi.get("<eos>", 2)

    cfg = ckpt["config"]
    model = CNNLSTM(
        num_embeddings=cfg["num_embeddings"],
        embedding_dim=cfg["embedding_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        padding_idx=cfg["padding_idx"],
        train_backbone=False,
        cnn_weights=None,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    image = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        toks = model.generate(image, start_token_id=bos_idx, end_token_id=eos_idx, max_len=args.max_len)[0]

    words = []
    for tid in toks.tolist():
        if tid == eos_idx:
            break
        if tid in (pad_idx, bos_idx):
            continue
        words.append(itos.get(tid, "<unk>"))

    print(" ".join(words))


if __name__ == "__main__":
    main()