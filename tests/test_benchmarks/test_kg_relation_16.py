"""Tests for the 16-class relation taxonomy."""
import pytest
from src.data.conceptnet import RELATION_CATEGORIES, categorize_relation
from src.benchmarks.conceptnet_tasks import RELATION_CLASSES
from src.benchmarks.benchmark_dataset import get_max_classes


def test_relation_categories_has_16_classes():
    assert len(RELATION_CATEGORIES) == 16


def test_synonym_and_antonym_are_separate():
    assert "Synonym" in RELATION_CATEGORIES
    assert "Antonym" in RELATION_CATEGORIES
    assert categorize_relation("Synonym") == "Synonym"
    assert categorize_relation("Antonym") == "Antonym"


def test_negation_class_exists():
    assert "Negation" in RELATION_CATEGORIES
    assert categorize_relation("NotCapableOf") == "Negation"
    assert categorize_relation("NotHasProperty") == "Negation"
    assert categorize_relation("NotDesires") == "Negation"


def test_formof_is_standalone():
    assert "FormOf" in RELATION_CATEGORIES
    assert categorize_relation("FormOf") == "FormOf"


def test_derivedfrom_is_standalone():
    assert "DerivedFrom" in RELATION_CATEGORIES
    assert categorize_relation("DerivedFrom") == "DerivedFrom"
    assert categorize_relation("EtymologicallyDerivedFrom") == "DerivedFrom"


def test_hascontext_is_standalone():
    assert "HasContext" in RELATION_CATEGORIES
    assert categorize_relation("HasContext") == "HasContext"


def test_hassubevent_is_standalone():
    assert "HasSubevent" in RELATION_CATEGORIES
    assert categorize_relation("HasSubevent") == "HasSubevent"
    assert categorize_relation("HasFirstSubevent") == "HasSubevent"
    assert categorize_relation("HasLastSubevent") == "HasSubevent"


def test_desire_class():
    assert "Desire" in RELATION_CATEGORIES
    assert categorize_relation("Desires") == "Desire"
    assert categorize_relation("MotivatedByGoal") == "Desire"


def test_no_other_class():
    """Other was removed — unmapped relations fall back to RelatedTo."""
    assert "Other" not in RELATION_CATEGORIES


def test_unknown_relation_falls_back_to_relatedto():
    assert categorize_relation("SomeUnknownRelation") == "RelatedTo"
    assert categorize_relation("FooBar") == "RelatedTo"


def test_all_34_raw_relations_mapped():
    """Every known raw relation maps to one of the 16 categories."""
    raw_rels = [
        "IsA", "DefinedAs", "MannerOf", "InstanceOf",
        "FormOf",
        "DerivedFrom", "EtymologicallyDerivedFrom", "EtymologicallyRelatedTo",
        "HasContext",
        "Synonym", "SimilarTo",
        "Antonym", "DistinctFrom",
        "RelatedTo", "SymbolOf",
        "UsedFor",
        "AtLocation", "LocatedNear",
        "PartOf", "HasA", "MadeOf",
        "HasSubevent", "HasFirstSubevent", "HasLastSubevent",
        "CapableOf", "ReceivesAction",
        "Causes", "HasPrerequisite", "CausesDesire", "Entails", "CreatedBy",
        "HasProperty",
        "Desires", "MotivatedByGoal",
        "NotCapableOf", "NotHasProperty", "NotDesires",
    ]
    for rel in raw_rels:
        cat = categorize_relation(rel)
        assert cat in RELATION_CATEGORIES, f"{rel} mapped to {cat} which is not in RELATION_CATEGORIES"


def test_relation_classes_matches_categories():
    assert len(RELATION_CLASSES) == 16
    assert set(RELATION_CLASSES) == set(RELATION_CATEGORIES)


def test_task_registry_kg_relation_16_classes():
    assert get_max_classes("kg_relation") == 16
