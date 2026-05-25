import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.cnn_lstm.model import CNNLSTM
from models.cnn_transformer.model import CNNTransformerDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="model.pt")
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--model-type", type=str, default="cnn_lstm", choices=["cnn_lstm", "cnn_transformer"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_config = ckpt["config"]
    model_type = ckpt_config.get("model_type")

    state_dict = ckpt["model_state_dict"]
    ckpt_max_len = ckpt_config.get("max_len")
    if ckpt_max_len is None and "pos_embed.weight" in state_dict:
        ckpt_max_len = state_dict["pos_embed.weight"].shape[0]

    if model_type is None:
        model_type = "cnn_transformer" if "nhead" in ckpt_config else "cnn_lstm"

    def build_model(model_type: str, config: dict | None = None):
        config = config or {}

        if model_type == "cnn_transformer":
            return CNNTransformerDecoder(
                num_embeddings=config["num_embeddings"],
                embedding_dim=config["embedding_dim"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                padding_idx=config["padding_idx"],
                train_backbone=False,
                cnn_weights=None,
                nhead=config["nhead"],
                max_len=ckpt_max_len if ckpt_max_len is not None else args.max_len
            ).to(device)

        return CNNLSTM(
                num_embeddings=config["num_embeddings"],
                embedding_dim=config["embedding_dim"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                padding_idx=config["padding_idx"],
                train_backbone=False,
                cnn_weights=None,
        ).to(device)

    stoi = ckpt["vocab_stoi"]
    itos = {v: k for k, v in stoi.items()}

    pad_idx = stoi.get("<pad>", 0)
    bos_idx = stoi.get("<bos>", 1)
    eos_idx = stoi.get("<eos>", 2)
    model = build_model(model_type, ckpt_config)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    #print("Loaded ckpt:", args.ckpt)
    #print("model_type:", model_type)
    #print("missing:", missing)
    #print("unexpected:", unexpected)
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