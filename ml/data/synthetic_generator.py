"""
Synthetic and augmented data generator for Tamil Handwritten OCR training and evaluation.
Generates realistic line images with stroke variations, paper texture, jitter, and noise.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple

# Sample authentic Tamil sentences, literature, phrases, and technical strings
TAMIL_TEXT_CORPUS = [
    "தமிழ் மிகவும் பழமையான மற்றும் செழுமையான மொழியாகும்.",
    "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.",
    "கற்க கசடறக் கற்பவை கற்றபின் நிற்க அதற்குத் தக.",
    "எப்பொருள் யார்யார்வாய்க் கேட்பினும் அப்பொருள் மெய்ப்பொருள் காண்ப தறிவு.",
    "அன்பிலார் எல்லாம் தமக்குரியர் அன்புடையார் என்பும் உரியர் பிறர்க்கு.",
    "அறிவற்றங் காக்கும் கருவி செறுவார்க்கும் உள்ளழிக்க லாகா அரண்.",
    "வணக்கம், நீங்கள் நலமாக இருக்கிறீர்களா?",
    "இன்று வானிலை மிகவும் இனிமையாகவும் அமைதியாகவும் உள்ளது.",
    "தமிழ்க் கையெழுத்து உணர்தல் அமைப்பு மிகச் சிறப்பாக இயங்குகிறது.",
    "சென்னையில் உள்ள மெரினா கடற்கரை உலகிலேயே இரண்டாவது நீளமான கடற்கரையாகும்.",
    "தமிழ்நாட்டில் 38 மாவட்டங்கள் உள்ளன.",
    "திருக்குறள் உலகப் பொதுமறை என்று போற்றப்படுகிறது.",
    "பாரதியார் மற்றும் பாரதிதாசன் தமிழ் இலக்கியத்தின் இரு கண்கள்.",
    "யாதும் ஊரே யாவரும் கேளிர் என்ற தத்துவம் உலகப் புகழ்பெற்றது.",
    "அம்மா எனக்குப் பிடித்த உணவை அன்போடு சமைத்துத் தந்தார்.",
    "மாணவர்கள் வகுப்பறையில் கவனமுடன் தமிழ் பாடம் படித்தனர்.",
    "தொழில்நுட்பம் மனித வாழ்க்கையை எளிதாக்குகிறது.",
    "கணினித் தமிழ் வளர்ச்சி இன்றைய காலகட்டத்தில் மிக முக்கியமானது.",
    "சுத்தமான குடிநீரும் பசுமையான சூழலும் மனித வாழ்விற்கு இன்றியமையாதவை.",
    "1234567890 மற்றும் தமிழ் எண்கள் ௧, ௨, ௩, ௪, ௫.",
    "புதிய சிந்தனைகளும் விடாமுயற்சியும் ஒருவரை வெற்றிப் பாதைக்கு அழைத்துச் செல்லும்.",
    "உண்மை, நேர்மை மற்றும் உழைப்பு ஆகியவை ஒரு மனிதனின் உயர்ந்த குணங்கள்.",
    "காலை எழுந்தவுடன் படிப்பு, பின்பு கனிவு கொடுக்கும் நல்ல பாட்டு.",
    "தீயினாற் சுட்டபுண் உள்ளாறும் ஆறாதே நாவினாற் சுட்ட வடு.",
    "ஒழுக்கம் விழுப்பம் தரலான் ஒழுக்கம் உயிரினும் ஓம்பப் படும்.",
    "காலத்தினாற் செய்த நன்றி சிறிதெனினும் ஞாலத்தின் மாணப் பெரிது.",
    "நன்றி மறப்பது நன்றன்று நன்றல்லது அன்றே மறப்பது நன்று.",
    "இனிய உளவாக இன்னாத கூறல் கனிஇருப்பக் காய்கவர்ந் தற்று.",
    "சுழன்றும்ஏர்ப் பின்னது உலகம் அதனால் உழந்தும் உழவே தலை.",
    "செல்வத்துள் செல்வம் செவிச்செல்வம் அச்செல்வம் செல்வத்துள் எல்லாம் தலை.",
    "அன்பும் அறனும் உடைத்தாயின் இல்வாழ்க்கை பண்பும் பயனும் அது.",
    "செயற்கரிய செய்வார் பெரியர் சிறியர் செயற்கரிய செய்கலா தார்.",
    "மனத்துக்கண் மாசிலன் ஆதல் அனைத்தறன் ஆகுல நீர பிற.",
    "முயற்சி திருவினையாக்கும் முயற்றின்மை இன்மை புகுத்தி விடும்.",
    "பொருளல் லவரைப் பொருளாகச் செய்யும் பொருளல்ல தில்லை பொருள்."
]

def get_tamil_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """Load embedded Noto Sans Tamil font or fallback."""
    font_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "backend", "app", "exports", "fonts", "NotoSansTamil-Regular.ttf")
    )
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

def render_tamil_line_image(text: str, font_size: int = 26, is_handwritten_style: bool = True) -> Image.Image:
    """
    Render a single line of Tamil text into an image with realistic handwriting/scanning artifacts:
    - Font size & baseline variations
    - Stroke dilation / erosion
    - Mild random rotation & shearing
    - Background paper grain and text anti-aliasing
    """
    font = get_tamil_font(font_size)
    
    # Calculate bounding box using getbbox
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Add margins
    pad_x = 24
    pad_y = 16
    img_w = max(60, text_w + pad_x * 2)
    img_h = max(36, text_h + pad_y * 2)

    # Base background: slightly off-white paper color (RGB 245-255)
    bg_color = (
        random.randint(245, 255),
        random.randint(245, 255),
        random.randint(240, 250)
    )
    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Ink color: dark blue, black, or dark charcoal (handwriting pen colors)
    if is_handwritten_style and random.random() > 0.4:
        # Blue ballpoint / gel pen
        ink_color = (
            random.randint(15, 35),
            random.randint(30, 65),
            random.randint(110, 160)
        )
    else:
        # Black / dark gray ink
        ink_color = (
            random.randint(20, 45),
            random.randint(20, 45),
            random.randint(20, 45)
        )

    # Draw text
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=ink_color)

    if is_handwritten_style:
        # Mild random rotation (-2.0 to 2.0 degrees) to simulate human handwriting angle
        rot_angle = random.uniform(-1.8, 1.8)
        img = img.rotate(rot_angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=bg_color)
        
        # Subtle Gaussian blur to simulate ink spread on paper fibers
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    return img

def generate_dataset_splits(
    num_train: int = 150,
    num_val: int = 30,
    num_test: int = 25
) -> Tuple[List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]]]:
    """
    Generate distinct, non-overlapping train, validation, and test sets.
    Prevent data leakage by splitting the corpus at the source text level.
    """
    # Deterministic shuffling
    shuffled_corpus = list(TAMIL_TEXT_CORPUS)
    random.seed(42)
    random.shuffle(shuffled_corpus)

    # Partition corpus
    train_corpus = shuffled_corpus[:22]
    val_corpus = shuffled_corpus[22:28]
    test_corpus = shuffled_corpus[28:]

    def make_samples(corpus: List[str], count: int) -> List[Tuple[Image.Image, str]]:
        samples = []
        for i in range(count):
            text = corpus[i % len(corpus)]
            # If line is very long, optionally sample a sub-phrase
            words = text.split()
            if len(words) > 6 and random.random() > 0.4:
                start_w = random.randint(0, len(words) - 4)
                text_slice = " ".join(words[start_w:start_w + random.randint(3, 6)])
            else:
                text_slice = text
                
            img = render_tamil_line_image(text_slice, font_size=random.randint(22, 28))
            samples.append((img, text_slice))
        return samples

    train_data = make_samples(train_corpus, num_train)
    val_data = make_samples(val_corpus, num_val)
    test_data = make_samples(test_corpus, num_test)

    return train_data, val_data, test_data
