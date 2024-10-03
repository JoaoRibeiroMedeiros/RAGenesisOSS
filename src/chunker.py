import re
import numpy as np
import pandas as pd


class Chunker:
    def __init__(self):
        self.texts = ["Bible", "Bible_NT", "Quran", "Torah", "Gita", "Analects"]

    def chunk_all(self):

        references_dict = {}
        verses_dict = {}

        for text in self.texts:
            references, verses = self.call_method_by_string(text)
            references_dict[text] = references
            verses_dict[text] = verses

        return references_dict, verses_dict

    def count_tokens(self, line):
        return len(re.findall(r"\w+", line))

    def call_method_by_string(self, method_name):
        # Create a mapping of strings to class methods
        methods = {
            "Bible": self.chunk_bible,
            "Bible_NT": self.chunk_new_testament,
            "Quran": self.chunk_quran,
            "Torah": self.chunk_torah,
            "Gita": self.chunk_gita,
            "Analects": self.chunk_analects,
        }

        # Call the method if it exists in the mapping
        method = methods.get(method_name)

        if method:
            return method()
        else:
            return "Invalid method name."

    def from_string_to_chunks(self, input_string):

        references_and_verses = {
            "Bible": (self.bible_verses_references, self.bible_verses),
            "Bible_NT": (self.new_testament_references, self.new_testament_verses),
            "Quran": (self.quran_verses_references, self.quran_verses),
            "Torah": (self.torah_verses_references, self.torah_verses),
            "Gita": (self.gita_verses_references, self.gita_verses),
            "Analects": (self.analects_verses_references, self.analects_verses),
        }

        return references_and_verses[input_string]

    def study_verses_length(self):

        verse_token_length_dict = {}
        for text in self.texts:
            references, verses = self.from_string_to_chunks(text)
            tokens_per_verse = [self.count_tokens(verse) for verse in verses]
            verse_token_length_dict[text] = tokens_per_verse

        verse_token_length_df = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in verse_token_length_dict.items()])
        )

        return verse_token_length_df

    def split_bible_and_torah(self, bible_verses, bible_verses_references):

        torah_verses = []
        torah_verses_references = []

        bible_verses_ = []
        bible_verses_references_ = []

        pentateuch_books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]

        for reference, verse in zip(bible_verses_references, bible_verses):
            # for book in pentateuch_books:
            torah = any(reference.split(" ")[0] in item for item in pentateuch_books)

            if torah:
                torah_verses.append(verse)
                torah_verses_references.append(reference)
            else:
                bible_verses_.append(verse)
                bible_verses_references_.append(reference)

        return (
            torah_verses_references,
            torah_verses,
            bible_verses_references_,
            bible_verses_,
        )

    def chunk_bible(self):

        file_path = "data/sacred_data/bible.txt"
        verses = []
        with open(file_path, "r") as file:
            for line in file:
                # Split the line at the first tab to separate the reference and the text
                reference, text = line.split("\t", 1)
                verses.append((reference.strip(), text.strip()))

        bible_verses = [verse[1] for verse in verses]
        bible_verses_references = [verse[0] for verse in verses]

        (
            torah_verses_references,
            torah_verses,
            bible_verses_references_,
            bible_verses_,
        ) = self.split_bible_and_torah(bible_verses, bible_verses_references)

        self.bible_verses = bible_verses
        self.bible_verses_references = bible_verses_references

        self.torah_verses = torah_verses
        self.torah_verses_references = torah_verses_references

        new_testament_verses = []
        new_testament_references = []

        for verse, reference in zip(bible_verses, bible_verses_references):
            if self.is_new_testament(reference):
                new_testament_verses.append(verse)
                new_testament_references.append(reference)

        self.new_testament_verses = new_testament_verses
        self.new_testament_references = new_testament_references

        return bible_verses_references_, bible_verses_

    def chunk_torah(self):

        return self.torah_verses_references, self.torah_verses

    def chunk_new_testament(self):

        return self.new_testament_references, self.new_testament_verses

    def chunk_quran(self):

        file_path = "data/sacred_data/quran.txt"
        verses = []
        with open(file_path, "r") as file:
            for line in file:
                # Split the line at the first tab to separate the reference and the text
                reference1, text = line.split("|", 1)
                reference2, text = text.split("|", 1)
                verses.append(
                    ("Surate " + reference1 + " verse " + reference2, text.strip())
                )
        quran_verses = [verse[1] for verse in verses]
        quran_verses_references = [verse[0] for verse in verses]

        self.quran_verses = quran_verses
        self.quran_verses_references = quran_verses_references

        return quran_verses_references, quran_verses

    def chunk_gita(self):
        file_path = "data/sacred_data/gita.txt"
        verses = []
        references = []
        in_translation_section = False
        with open(file_path, "r") as file:
            for line in file:
                # Split the line at the first tab to separate the reference and the text
                if "- CHAPTER" in line:
                    matches = re.findall(r"- CHAPTER (\d+)", line)
                    # print(matches)
                    reference_chapter = matches[0]
                    reference_verse_number = 0

                if "TRANSLATION" in line:
                    verse = ""
                    reference_verse_number = reference_verse_number + 1
                    in_translation_section = True
                    continue

                # Check if the current line contains "PURPORT"
                if "PURPORT" in line:
                    references.append(
                        "Chapter "
                        + str(reference_chapter)
                        + " Verse "
                        + str(reference_verse_number)
                    )
                    verse = verse.replace("\n", " ")
                    verse = verse.replace(
                        "Copyright © 1998 The Bhaktivedanta Book Trust Int'l. All Rights Reserved.",
                        "",
                    )
                    verses.append(verse)
                    in_translation_section = False
                    continue

                if in_translation_section:
                    verse = verse + line

        self.gita_verses = verses
        self.gita_verses_references = references

        return references, verses

    def add_analect_chapter_name(self, reference):
        analects_chapters = {
            "1": "Xue er 學而",
            "2": "Wei zheng 爲政",
            "3": "Ba yi 八佾",
            "4": "Li ren 里仁",
            "5": "Gongye Chang 公冶長",
            "6": "Yong ye 雍也",
            "7": "Shu er 述而",
            "8": "Tai Bo 泰伯",
            "9": "Zi nan 子罕",
            "10": "Xiang dang 鄕黨",
            "11": "Xianjin 先進",
            "12": "Yan Yuan 顏淵",
            "13": "Zi Lu 子路",
            "14": "Xian wen 憲問",
            "15": "Wei Ling Gong 衞靈公",
            "16": "Ji shi 季氏",
            "17": "Yang Huo 陽貨",
            "18": "Weizi 微子",
            "19": "Zi Zhang 子張",
            "20": "Yao yue 堯曰",
        }
        reference_ = reference
        chapter = reference_.split("[")[1].split(":")[0]
        chapter_name = analects_chapters[chapter]
        reference_ = chapter_name + " " + reference_
        return reference_

    def chunk_analects(self):

        # Split the input string into lines
        from data.sacred_data.analects import analects

        lines = analects.split("\n")

        verses = []
        references = []

        for line in lines:
            # print(line)
            if len(line) > 1:
                if line[0] == "[" and line[2] == "-" and line[4] == "]":
                    continue  # Skip lines matching [n-m]
                elif line[0] == "[" and line[3] == "-" and line[5] == "]":
                    continue  # Skip lines matching [n-m]
                elif line[0] == "[" and line[2] == "-" and line[5] == "]":
                    continue  # Skip lines matching [n-m]
                elif line[0] == "[" and line[3] == "-" and line[6] == "]":
                    continue  # Skip lines matching [n-m]
                elif line[0] == "[" and line[2] == ":" and line[4] == "]":
                    reference = self.add_analect_chapter_name(line[:5])
                    references.append(reference)
                    verses.append(line[5:])  # Collect lines starting with n.
                elif line[0] == "[" and line[3] == ":" and line[5] == "]":
                    reference = self.add_analect_chapter_name(line[:6])
                    references.append(reference)
                    verses.append(line[6:])  # Collect lines starting with n.
                elif line[0] == "[" and line[2] == ":" and line[5] == "]":
                    reference = self.add_analect_chapter_name(line[:6])
                    references.append(reference)
                    verses.append(line[6:])  # Collect lines starting with n.
                elif line[0] == "[" and line[3] == ":" and line[6] == "]":
                    reference = self.add_analect_chapter_name(line[:7])
                    references.append(reference)
                    verses.append(line[7:])  # Collect lines starting with n.
                elif line[0:9] == "[Comment]":
                    continue
                elif line[1] == ".":
                    continue  # Skip lines matching n.
                elif line[2] == ".":
                    continue  # Skip lines matching n.
                else:
                    references.append(reference)
                    verses.append(line)  # Collect other lines
            else:
                continue

        cleaned_verses = [s for s in verses if s.strip()]
        cleaned_references = [
            references[i] for i in range(len(verses)) if verses[i].strip()
        ]

        cleaned_references, cleaned_verses = self.filter_lists(
            cleaned_references, cleaned_verses
        )

        self.analects_verses = cleaned_verses
        self.analects_verses_references = cleaned_references

        return cleaned_references, cleaned_verses

    def is_new_testament(self, reference):
        # List of New Testament books
        new_testament_books = [
            "Matthew",
            "Mark",
            "Luke",
            "John",
            "Acts",
            "Romans",
            "1 Corinthians",
            "2 Corinthians",
            "Galatians",
            "Ephesians",
            "Philippians",
            "Colossians",
            "1 Thessalonians",
            "2 Thessalonians",
            "1 Timothy",
            "2 Timothy",
            "Titus",
            "Philemon",
            "Hebrews",
            "James",
            "1 Peter",
            "2 Peter",
            "1 John",
            "2 John",
            "3 John",
            "Jude",
            "Revelation",
        ]

        for book in new_testament_books:
            if book in reference:
                return True
        return False

    def filter_lists(self, list1, list2):
        # Track the first occurrence of each element in list1
        seen = {}
        repeated = set()

        # Determine repeated elements in list1
        for index, item in enumerate(list1):
            if item in seen:
                repeated.add(item)
            else:
                seen[item] = index

        # Compile unique elements to keep their first occurrence only
        list1_unique = []
        seen_in_result = set()

        for item in list1:
            if item in repeated:
                if item not in seen_in_result:
                    list1_unique.append(item)
                    seen_in_result.add(item)
            else:
                list1_unique.append(item)

        # List2 should correspond to the non-repeated indices in list1
        list2_filtered = [list2[i] for i in seen.values() if list1[i] in list1_unique]

        return list1_unique, list2_filtered


# %%


# # Example usage
# list1 = [1, 2, 2, 3, 4, 4, 5]
# list2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# list1_result, list2_result = filter_lists(list1, list2)

# print("Filtered List 1:", list1_result)
# print("Filtered List 2:", list2_result)


# %%
