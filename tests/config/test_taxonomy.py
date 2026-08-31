from cropai.domain.taxonomy import Taxonomy


def test_taxonomy_integrity():
    errors = Taxonomy().validate_integrity()
    assert errors == []


def test_counts_are_nonzero():
    tax = Taxonomy()
    assert len(tax.crop_ids()) >= 17
    assert len(tax.disease_ids()) >= 30
    assert len(tax.pest_ids()) >= 10
    assert len(tax.healthy_ids()) >= 17


def test_severity_bins_cover_0_100():
    tax = Taxonomy()
    assert tax.severity_bins[0].min_pct == 0.0
    assert tax.severity_bins[-1].max_pct == 100.0
    assert tax.severity_from_area(0).id == "healthy"
    assert tax.severity_from_area(8).id == "early"
    assert tax.severity_from_area(20).id == "moderate"
    assert tax.severity_from_area(40).id == "severe"
    assert tax.severity_from_area(80).id == "very_severe"
    assert tax.severity_from_area(100).id == "very_severe"


def test_maharashtra_operational_crops_include_cropsap():
    tax = Taxonomy()
    ops = set(tax.operational_crops())
    for crop in ["rice", "cotton", "soybean", "sugarcane", "maize", "pigeon_pea", "sorghum", "chickpea"]:
        assert crop in ops


def test_top10_indian_crops_are_present_and_operational():
    tax = Taxonomy()
    top10 = tax.top10_india()
    assert top10 == [
        "rice",
        "wheat",
        "sugarcane",
        "cotton",
        "maize",
        "soybean",
        "chickpea",
        "groundnut",
        "mustard",
        "pearl_millet",
    ]
    ops = set(tax.operational_crops())
    for crop in top10:
        assert crop in tax.crop_ids()
        assert crop in ops


def test_tomato_potato_maize_grape_are_first_class():
    tax = Taxonomy()
    required = ["tomato", "potato", "maize", "grape"]
    assert tax.horticulture_required() == required
    ops = set(tax.operational_crops())
    for crop in required:
        assert crop in tax.crop_ids()
        assert crop in ops
        assert "horticulture_required" in list(tax.crops[crop].get("tiers") or [])


def test_list_was_not_narrowed():
    tax = Taxonomy()
    # Existing Maharashtra / horticulture crops must remain.
    for crop in ["onion", "pepper", "pigeon_pea", "sorghum"]:
        assert crop in tax.crop_ids()


def test_no_local_field_images_are_claimed():
    tax = Taxonomy()
    missing = tax.crops_missing_local_field()
    assert set(missing) == set(tax.crop_ids())


def test_confidence_thresholds_are_conservative():
    tax = Taxonomy()
    assert tax.confidence["diagnosis_high"] >= 0.8
    assert tax.confidence["expert_referral_below"] <= tax.confidence["diagnosis_review"]
