"""Vote-K picks budget-many items that spread over the data instead of crowding into one topic."""

import numpy as np

from purpose_coder.data import Project
from purpose_coder.votek import select_for_annotation, vote_k
from tests.fakes import fake_embed

TOPICS = {
    "vaccine": "vaccinate children against measles polio in village clinic number {i}",
    "building": "construct hospital building with wards and a new roof for site {i}",
    "training": "train nurses and midwives in a course for health workers batch {i}",
}


def synthetic_projects(per_topic=8):
    projects, topic_of = [], []
    for topic, template in TOPICS.items():
        for i in range(per_topic):
            projects.append(Project(topic.title(), template.format(i=i), "122"))
            topic_of.append(topic)
    return projects, topic_of


def test_three_picks_cover_three_topics():
    projects, topic_of = synthetic_projects()
    picked = select_for_annotation(projects, budget=3, embed=fake_embed, k=5)
    assert len(picked) == 3
    assert {topic_of[i] for i in picked} == set(TOPICS)


def test_six_picks_are_distinct_and_budget_many():
    projects, _ = synthetic_projects()
    picked = select_for_annotation(projects, budget=6, embed=fake_embed, k=5)
    assert len(picked) == len(set(picked)) == 6


def test_picks_are_diverse_not_just_the_densest_topic():
    projects, topic_of = synthetic_projects(per_topic=8)
    # Make the vaccine topic much bigger: plain "most central first" would take its items one after another.
    extra = [Project("Vaccine", TOPICS["vaccine"].format(i=100 + i), "122") for i in range(16)]
    picked = select_for_annotation(projects + extra, budget=3, embed=fake_embed, k=5)
    topics = [(topic_of + ["vaccine"] * 16)[i] for i in picked]
    assert set(topics) == set(TOPICS)


def test_budget_larger_than_the_data_returns_everything_once():
    embeddings = np.eye(4)
    assert sorted(vote_k(embeddings, budget=10, k=2)) == [0, 1, 2, 3]


def test_zero_budget_and_empty_input_pick_nothing():
    assert vote_k(np.eye(3), budget=0) == []
    assert select_for_annotation([], budget=5, embed=fake_embed) == []


def test_the_same_input_gives_the_same_picks():
    projects, _ = synthetic_projects()
    first = select_for_annotation(projects, budget=5, embed=fake_embed, k=5)
    assert first == select_for_annotation(projects, budget=5, embed=fake_embed, k=5)
