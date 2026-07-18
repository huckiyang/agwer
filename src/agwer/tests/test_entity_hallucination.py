"""entity_f1 and word_hallucination_rate: hand-computed cases, including the
repetition-hallucination mechanism from arXiv:2408.16180 Table 7."""

import pytest

import agwer

# ------------------------------------------------------------- entity_f1

def test_entity_errors_hide_inside_acceptable_wer():
    ref = "okay so please transfer five hundred dollars to the savings account before friday and send me a confirmation text right away"
    hyp = "okay so please transfer nine hundred dollars to the savings account before friday and send me a confirmation text right away"
    assert agwer.wer(ref, hyp) == pytest.approx(1 / 21)   # 4.8%, looks fine
    m = agwer.entity_f1(ref, hyp, entities={"five", "hundred"})
    assert m["recall"] == pytest.approx(0.5)              # it is not fine
    assert m["entity_wer"] == pytest.approx(0.5)


def test_entity_f1_hand_computed():
    ref = "send five hundred dollars to acme"
    hyp = "send five hundred dollars to ackme"
    m = agwer.entity_f1(ref, hyp, entities={"five", "hundred", "dollars", "acme", "ackme"})
    assert m["ref_entities"] == 4 and m["hyp_entities"] == 4
    assert m["recall"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)


def test_numeric_tokens_predicate():
    m = agwer.entity_f1(
        "the code is seven two nine and the total is 42 dollars",
        "the code is seven two five and the total is 42 dollars",
        predicate=agwer.numeric_tokens,
    )
    assert m["ref_entities"] == 4
    assert m["recall"] == pytest.approx(0.75)


def test_entity_f1_selector_validation():
    with pytest.raises(ValueError):
        agwer.entity_f1("a", "a")
    with pytest.raises(ValueError):
        agwer.entity_f1("a", "a", entities={"a"}, predicate=str.isdigit)


def test_entity_f1_none_when_no_entities():
    m = agwer.entity_f1("hello there", "hello there", entities={"acme"})
    assert m["recall"] is None and m["f1"] is None


# ------------------------------------------- word_hallucination_rate (WHR)

def test_repetition_hallucination_autoregressive_loop():
    # the arXiv:2408.16180 Table 7 mechanism: the model loops, copying words
    # the ASR really produced, more times than they were heard. A vocabulary
    # check sees nothing wrong; occurrence-bounded attribution does.
    ref = "the show was great"
    onebest = "the show was great"
    looped = "the show was great the show was great the show was great"
    m = agwer.word_hallucination_rate([ref], [looped], [onebest])
    assert m["repetition_hallucinations"] == 8      # two full spurious copies
    assert m["novel_hallucinations"] == 0
    assert m["whr"] == pytest.approx(8 / 12)


def test_novel_hallucination_invented_word():
    ref = "schedule the demo for tuesday"
    onebest = "schedule the demo for tuesday"
    out = "schedule the ultrasonic demo for tuesday"
    m = agwer.word_hallucination_rate([ref], [out], [onebest])
    assert m["novel_hallucinations"] == 1
    assert m["repetition_hallucinations"] == 0
    assert m["whr"] == pytest.approx(1 / 6)


def test_passed_through_asr_error_is_not_hallucination():
    # copying the (wrong) input is an ASR error, not a hallucination
    ref = "meet me at the dock"
    onebest = "meet me at the duck"
    m = agwer.word_hallucination_rate([ref], [onebest], [onebest])
    assert m["hallucinated_tokens"] == 0
    assert m["passed_through_errors"] == 1
    assert m["whr"] == 0.0


def test_generative_recovery_not_penalized():
    # correct tokens absent from every hypothesis are the rho>1 mechanism
    ref = "run uv pip install agwer now"
    nbest = [["run you v pip install ag where now", "run uv pip install a g wear now"]]
    m = agwer.word_hallucination_rate([ref], [ref], nbest)
    assert m["whr"] == 0.0
    assert m["generative_tokens"] >= 1              # 'agwer' recovered
    assert m["hallucinated_tokens"] == 0


def test_whr_accepts_onebest_and_nbest():
    ref, hyp = "a b c", "a b c"
    one = agwer.word_hallucination_rate([ref], [hyp], [hyp])          # 1-best str
    many = agwer.word_hallucination_rate([ref], [hyp], [[hyp, "a b"]])  # n-best
    assert one["whr"] == many["whr"] == 0.0


def test_correct_repetition_of_heard_word_not_penalized():
    # 'the' appears twice in both reference and output; budget from the
    # 1-best is two: nothing is hallucinated.
    ref = "the cat saw the dog"
    onebest = "the cat saw the dog"
    m = agwer.word_hallucination_rate([ref], [ref], [onebest])
    assert m["whr"] == 0.0 and m["hallucinated_tokens"] == 0
