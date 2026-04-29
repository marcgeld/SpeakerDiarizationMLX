"""
Generate multilingual test conversations for SpeakerDiarizationMLX.

Creates short two-speaker dialogues in English, Swedish, German, and French
using Google Text-to-Speech (gTTS). Each conversation contains six lines.
Outputs individual WAV files in sample_audio/.
"""

from gtts import gTTS
from pydub import AudioSegment
from pathlib import Path

# Output directory
out_dir = Path("sample_audio")
out_dir.mkdir(parents=True, exist_ok=True)

# Define multilingual dialogues (6 lines each)
dialogs = {
    "english": [
        ("en", "male", "Hi, how are you today?"),
        ("en", "female", "I'm great, thanks! Did you get the project update?"),
        ("en", "male", "Yes, I did. The results look fantastic."),
        ("en", "female", "Awesome! Should we schedule a review meeting?"),
        ("en", "male", "Sure, tomorrow morning works for me."),
        ("en", "female", "Perfect, see you then."),
    ],
    "swedish": [
        ("sv", "male", "Hej, hur mår du idag?"),
        ("sv", "female", "Jag mår bra tack! Har du sett rapporten än?"),
        ("sv", "male", "Ja, jag läste den i morse."),
        ("sv", "female", "Vad tyckte du om resultatet?"),
        ("sv", "male", "Det ser lovande ut, men vi behöver fler tester."),
        ("sv", "female", "Helt klart, vi kör imorgon igen."),
    ],
    "german": [
        ("de", "male", "Hallo, wie geht es dir heute?"),
        ("de", "female", "Mir geht's gut, danke. Hast du die neuen Zahlen gesehen?"),
        ("de", "male", "Ja, sie sehen besser aus als letzte Woche."),
        ("de", "female", "Super! Dann können wir bald abschließen."),
        ("de", "male", "Genau, ich melde mich später mit Details."),
        ("de", "female", "In Ordnung, bis später!"),
    ],
    "french": [
        ("fr", "male", "Salut, comment ça va aujourd'hui?"),
        ("fr", "female", "Très bien merci! As-tu vu le rapport final?"),
        ("fr", "male", "Oui, il est impressionnant."),
        ("fr", "female", "Parfait! On en discute demain matin?"),
        ("fr", "male", "D'accord, à neuf heures?"),
        ("fr", "female", "Oui, à demain!"),
    ],
}

# Generate and export each conversation
for lang_name, lines in dialogs.items():
    clips = []
    for lang, voice, text in lines:
        tts = gTTS(text=text, lang=lang)
        tmp_file = out_dir / f"tmp_{voice}_{lang_name}.mp3"
        tts.save(tmp_file)
        seg = AudioSegment.from_mp3(tmp_file)
        clips.append(seg)
        clips.append(AudioSegment.silent(duration=600))  # short pause

    conversation = sum(clips)
    out_path = out_dir / f"conversation_{lang_name}.wav"
    conversation.export(out_path, format="wav")
    print(f"Saved {out_path}")

print("All multilingual conversations generated successfully.")
