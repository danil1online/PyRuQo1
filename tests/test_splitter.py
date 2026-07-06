import pytest
import sys
from unittest.mock import MagicMock, patch
import tempfile
import os

sys.modules['fitz'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['tokenizers'] = MagicMock()
sys.modules['bitsandbytes'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['marker'] = MagicMock()
sys.modules['marker.models'] = MagicMock()
sys.modules['marker.convert'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()
sys.modules['llama_cpp.convert'] = MagicMock()


def test_is_article_start_udk():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    assert splitter._is_article_start("УДК 517.5") is True
    assert splitter._is_article_start("УДК 621.3") is True


def test_is_article_start_abstract():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    assert splitter._is_article_start("Аннотация") is True
    assert splitter._is_article_start("Abstract") is True
    assert splitter._is_article_start("Введение") is True


def test_is_article_start_normal():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    assert splitter._is_article_start("Просто текст статьи") is False
    assert splitter._is_article_start("Lorem ipsum dolor sit amet") is False


def test_split_folder_not_found():
    from pyruqo1.dataset.splitter import JournalSplitter
    splitter = JournalSplitter()
    result = splitter.split_folder("/nonexistent/path")
    assert result == []


def test_force_split_enabled_when_article_count_is_1_and_pages_gt_8():
    """Если найдено ≤1 статья и страниц >8, должен сработать fallback."""
    from pyruqo1.dataset.splitter import JournalSplitter

    with tempfile.TemporaryDirectory() as tmpdir:
        test_pdf = os.path.join(tmpdir, "test_journal.pdf")
        open(test_pdf, "w").close()

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=20)

        mock_pages = []
        for i in range(20):
            mock_page = MagicMock()
            if i == 0:
                mock_page.get_text.return_value = "Титульная страница"
            else:
                mock_page.get_text.return_value = "Просто текст без маркеров статей"
            mock_pages.append(mock_page)
        mock_doc.__getitem__.side_effect = lambda idx: mock_pages[idx]
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))

        mock_new_doc = MagicMock()

        with patch("pyruqo1.dataset.splitter.fitz.open") as mock_fitz_open:
            mock_fitz_open.side_effect = [mock_doc] + [mock_new_doc] * 6

            splitter = JournalSplitter()
            splitter.split_pdf(test_pdf, output_dir=tmpdir)

        # Fallback: 20 страниц / 4 = 5 файлов → 5 вызовов save
        assert mock_new_doc.save.call_count == 5


def test_force_split_disabled_when_article_count_gt_1():
    """Если найдено >1 статьи, fallback не срабатывает."""
    from pyruqo1.dataset.splitter import JournalSplitter

    with tempfile.TemporaryDirectory() as tmpdir:
        test_pdf = os.path.join(tmpdir, "test_journal2.pdf")
        open(test_pdf, "w").close()

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=12)

        mock_pages = []
        for i in range(12):
            mock_page = MagicMock()
            if i == 0:
                mock_page.get_text.return_value = "Титульная страница"
            elif i == 4:
                mock_page.get_text.return_value = "УДК 517.5\nАннотация текста статьи"
            elif i == 8:
                mock_page.get_text.return_value = "УДК 621.3\nАннотация второй статьи"
            else:
                mock_page.get_text.return_value = "Просто текст"
            mock_pages.append(mock_page)
        mock_doc.__getitem__.side_effect = lambda idx: mock_pages[idx]
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))

        mock_new_doc = MagicMock()

        with patch("pyruqo1.dataset.splitter.fitz.open") as mock_fitz_open:
            mock_fitz_open.side_effect = [mock_doc] + [mock_new_doc] * 7

            splitter = JournalSplitter()
            splitter.split_pdf(test_pdf, output_dir=tmpdir)

        # 3 статьи → 3 вызова save (без force_)
        assert mock_new_doc.save.call_count == 3


def test_force_split_not_triggered_when_pages_lte_8():
    """Если страниц ≤8, fallback не срабатывает даже при 1 статье."""
    from pyruqo1.dataset.splitter import JournalSplitter, FORCE_PAGES_PER_ARTICLE

    assert FORCE_PAGES_PER_ARTICLE == 4

    with tempfile.TemporaryDirectory() as tmpdir:
        test_pdf = os.path.join(tmpdir, "test_journal3.pdf")
        open(test_pdf, "w").close()

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=8)

        mock_pages = []
        for i in range(8):
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Просто текст без маркеров"
            mock_pages.append(mock_page)
        mock_doc.__getitem__.side_effect = lambda idx: mock_pages[idx]
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))

        mock_new_doc = MagicMock()

        with patch("pyruqo1.dataset.splitter.fitz.open") as mock_fitz_open:
            mock_fitz_open.side_effect = [mock_doc, mock_new_doc, mock_new_doc]

            splitter = JournalSplitter()
            splitter.split_pdf(test_pdf, output_dir=tmpdir)

        # pages=8, ≤1 статья, но pages <= 8 → fallback не срабатывает
        # 1 статья (все 8 страниц) → 1 save (>= 2 страниц)
        assert mock_new_doc.save.call_count == 1


def test_force_split_correct_page_ranges():
    """Проверяем, что fallback режет по 4 страницы."""
    from pyruqo1.dataset.splitter import JournalSplitter

    with tempfile.TemporaryDirectory() as tmpdir:
        test_pdf = os.path.join(tmpdir, "test_journal4.pdf")
        open(test_pdf, "w").close()

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=20)

        mock_pages = []
        for i in range(20):
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Просто текст"
            mock_pages.append(mock_page)
        mock_doc.__getitem__.side_effect = lambda idx: mock_pages[idx]
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))

        mock_new_doc = MagicMock()

        with patch("pyruqo1.dataset.splitter.fitz.open") as mock_fitz_open:
            mock_fitz_open.side_effect = [mock_doc] + [mock_new_doc] * 10

            splitter = JournalSplitter()
            splitter.split_pdf(test_pdf, output_dir=tmpdir)

        # Проверяем аргументы insert_pdf — должны быть диапазоны по 4 страницы
        insert_calls = mock_new_doc.insert_pdf.call_args_list
        assert len(insert_calls) == 5

        # Ожидаемые диапазоны: from_page=0, to_page=3; from_page=4, to_page=7; и т.д.
        expected_ranges = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19)]
        for i, (_, kwargs) in enumerate(insert_calls):
            assert kwargs["from_page"] == expected_ranges[i][0]
            assert kwargs["to_page"] == expected_ranges[i][1]
