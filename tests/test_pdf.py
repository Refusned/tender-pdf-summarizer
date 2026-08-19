import io

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.pdf import (
    NoTextLayerError,
    PdfExtractionError,
    extract_pages,
    render_for_llm,
    select_pages,
)


def test_извлекает_страницы_с_кириллицей(sample_pdf):
    pages = extract_pages(sample_pdf)

    assert len(pages) == 5
    assert "НМЦК" in pages[0].text
    assert pages[0].number == 1


def test_битый_файл_отклоняется():
    with pytest.raises(PdfExtractionError):
        extract_pages(b"%PDF-1.4 broken bytes without xref")


def test_pdf_без_текстового_слоя_отклоняется():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.showPage()
    pdf.save()

    with pytest.raises(NoTextLayerError):
        extract_pages(buffer.getvalue())


def test_при_нехватке_бюджета_страница_со_штрафами_важнее_середины(sample_pdf):
    pages = extract_pages(sample_pdf)
    # Бюджета хватает на обязательные первые две страницы и ещё одну.
    budget = len(pages[0].text) + len(pages[1].text) + len(pages[4].text) + 10

    numbers = [p.number for p in select_pages(pages, budget)]

    assert numbers == [1, 2, 5], "ожидали титул, оглавление и раздел про неустойку"


def test_разметка_для_модели_нумерует_страницы(sample_pdf):
    pages = select_pages(extract_pages(sample_pdf), 100_000)

    document = render_for_llm(pages, 100_000)

    assert "=== Страница 1 ===" in document
    assert "=== Страница 5 ===" in document
    assert document.index("Страница 1") < document.index("Страница 5")


def test_бюджет_символов_соблюдается(sample_pdf):
    pages = extract_pages(sample_pdf)

    document = render_for_llm(pages, 300)

    assert len(document) <= 300
