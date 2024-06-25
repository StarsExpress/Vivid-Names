from configs.names_config import EMBED_CHARS_DICT
from utils.files_helper import read_unique_names
from utils.embeddings import embed_name


start_char = EMBED_CHARS_DICT['start']
end_char = EMBED_CHARS_DICT['end']


def test_embeddings(name_type: str):
    original_names = read_unique_names(name_type)
    print(f'read_unique_names test passed.\n')

    embedded_names = [embed_name(name) for name in original_names]
    for original_name, embedded_name in zip(original_names, embedded_names):
        assert embedded_name[0] == start_char
        assert embedded_name[-1] == end_char
        assert len(embedded_name) == len(original_name) + 2
        assert set(original_name) & {start_char, end_char} == set()

    print(f'{name_type} embeddings test passed.\n')


if __name__ == '__main__':
    test_embeddings('surnames')
    test_embeddings('female_forenames')
    test_embeddings('male_forenames')
