import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AdamW
from model import load_model_and_tokenizer
from dataset import PreferenceDataset
from loss import dpo_loss
from clean_dataset import clean_raw_data
from copy import deepcopy
import matplotlib.pyplot as plt

def main():
    """
    Main function to execute the training loop for DPO.
    """

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a language model using Direct Preference Optimization.')
    parser.add_argument('--model_name', type=str, default='gpt2', help='The name of the pre-trained language model.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10, help='The number of training epochs.')
    parser.add_argument('--beta', type=float, default=0.1, help='The hyperparameter beta for DPO.')
    parser.add_argument('--batch_size', type=int, default=4, help='The batch size for training.')
    parser.add_argument('--train_split', type=float, default=0.6, help='The fraction of data to use for training.')
    parser.add_argument('--max_length', type=int, default=512, help='The maximum sequence length for tokenization.')
    parser.add_argument('--data_path', type=str, default='data/instruction-data-with-preference.json', help='The path to the raw preference data.')
    parser.add_argument('--train_data_path', type=str, default='data/train_data.json', help='The path to save the cleaned training data.')
    parser.add_argument('--test_data_path', type=str, default='data/test_data.json', help='The path to save the cleaned testing data.')
    parser.add_argument('--output_dir', type=str, default='output_model', help='The directory to save the fine-tuned model.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='The device to run the training on.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Model configuration
    MODEL_NAME = args.model_name
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)

    # Create a reference model for DPO
    ref_model = deepcopy(model)
    ref_model.to(device)
    ref_model.eval()

    # Exclude reference model parameters from optimization
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Load the raw preference data
    train_data, test_data = clean_raw_data(args.data_path, args.train_data_path, args.test_data_path, args.train_split)
    print(f"Loaded {len(train_data)} preference pairs.")
    print("Example:")
    print(train_data[0])

    # Dataset and DataLoader configuration
    dataset = PreferenceDataset(train_data, tokenizer, max_length=args.max_length)
    test_dataset = PreferenceDataset(test_data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Compute the DPO loss
            loss = dpo_loss(model, ref_model, batch, args.beta, device)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Logging
            if (step + 1) % 10 == 0 or (step + 1) == len(dataloader):
                avg_loss = total_loss / (step + 1)
                print(f"  Step {step + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for test_batch in test_dataloader:
                test_loss += dpo_loss(model, ref_model, test_batch, args.beta, device).item()
            avg_test_loss = test_loss / len(test_dataloader)

        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1} completed. Average training Loss: {avg_loss:.4f}. Average test Loss: {avg_test_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    # Plot the training and validation loss
    epochs = range(1, args.num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Testing Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

if __name__ == '__main__':
    main()
