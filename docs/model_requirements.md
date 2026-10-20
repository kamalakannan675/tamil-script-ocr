# Model Requirements and Inventory: TamilScript OCR

## 1. Character Inventory & Vocabulary

TamilScript OCR supports the complete Tamil Unicode script specification (`U+0B80` – `U+0BFF`), Grantha loan consonants, digits, and standard punctuation.

| Category | Count | Characters |
|---|---|---|
| **CTC Blank** | 1 | `<blank>` (Index 0) |
| **Vowels (உயிர்)** | 13 | அ, ஆ, இ, ஈ, உ, ஊ, எ, ஏ, ஐ, ஒ, ஓ, ஔ, ஃ (Aytham) |
| **Consonants (மெய் base)** | 18 | க, ங, ச, ஞ, ட, ண, த, ந, ப, ம, ய, ர, ல, வ, ழ, ள, ற, ன |
| **Grantha Consonants** | 6 | ஜ, ஷ, ஸ, ஹ, க்ஷ, ஸ்ரீ |
| **Dependent Vowel Signs** | 12 | ் (Pulli / Virama), ா, ி, ீ, ு, ூ, ெ, ே, ை, ொ, ோ, ௌ |
| **Tamil Numerals** | 13 | ௦, ௧, ௨, ௩, ௪, ௫, ௬, ௭, ௮, ௯, ௰, ௱, ௲ |
| **ASCII & Punctuation** | 24 | Space, 0–9, `.`, `,`, `!`, `?`, `-`, `:`, `;`, `'`, `"`, `(`, `)`, `/`, `%` |
| **Total Vocabulary** | **87** | All indexed deterministically in `backend/app/ocr/unicode_utils.py` |

## 2. Checkpoint Details

- **Default Checkpoint Path:** `backend/models/tamil_crnn.pt`
- **Alternative Checkpoint Path:** `ml/checkpoints/tamil_crnn.pt`
- **File Format:** PyTorch `.pt` containing `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_cer`, `vocab_size`.
- **Parameter Count:** ~8.5 Million parameters.
- **Weights File Size:** ~34.2 MB.
- **Memory Footprint at Inference:** ~85 MB RAM.
- **Hardware Requirement:** Modest CPU (Intel Core i3 / AMD Ryzen 3 or higher, 512 MB – 1 GB available RAM).

## 3. Alternative Transformer Model (Optional Adapter)

- **Model Name:** `sabaridsnfuji/Tamil_Offline_Handwritten_OCR`
- **Architecture:** Vision Encoder-Decoder (`google/vit-base-patch16-224-in21k` + `d42kw01f/Tamil-RoBERTa`)
- **License:** MIT License
- **Adapter Location:** `backend/app/ocr/recognition.py` (`VisionEncoderDecoderTamilRecognizer`)
- **Resource Requirement:** ~2 GB RAM, 640 MB disk.
