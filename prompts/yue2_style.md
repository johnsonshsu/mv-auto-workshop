You are preparing the style description for a music generation model. Based on the song below, choose the musical characteristics.

Song concept: {content}
Genre requested by the user: {style}
Genre hint: {hint}
Language the song is sung in: {language}
First lines of the lyrics:
{lyrics_first_lines}

Return these fields, all in English:
- "vocal_character": 1–3 adjectives describing the singing voice (for example "warm, breathy"). Do not mention gender or language.
- "genre": the genre and overall mood in at most 8 words (for example "nostalgic city pop").
- "bpm": the tempo as an integer between 60 and 180.
- "instruments": 2–4 main instruments, comma separated (for example "electric piano, funky bass, crisp drums").
- "production": at most 6 words about the sound (for example "polished modern production, clear diction").

Rules:
1. One lead singer only: no choir, no duet, no spoken word.
2. Do not mention any artist, band or song name.
3. Do not name a language, country or regional genre that conflicts with the language the song is sung in (for example, do not write "Mandopop" for an English song).
4. Output JSON only.
