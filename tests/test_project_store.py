from __future__ import annotations

import json

import pytest

from app.core.models import DONE, FAILED, PENDING, RUNNING, STAGE_IDS, STALE, UserFacingError
from app.storage.project_store import SUBDIRS, ProjectStore


def test_create_builds_folder_structure_and_project_file(tmp_path):
    project = ProjectStore(tmp_path).create("夏夜的風", {"engine": "infinitetalk"})
    for sub in ("input", "lyrics", "audio", "analysis", "segments", "output", "logs"):
        assert (project.folder / sub).is_dir()
    data = json.loads((project.folder / "project.json").read_text(encoding="utf-8"))
    assert data["name"] == "夏夜的風"
    assert data["inputs"]["engine"] == "infinitetalk"
    assert set(data["stages"]) == set(STAGE_IDS)
    assert not (project.folder / "project.json.tmp").exists()


@pytest.mark.parametrize("name", ["", "   ", "a/b", "a:b", "what?", "結尾有點."])
def test_create_rejects_names_that_cannot_be_folders(tmp_path, name):
    with pytest.raises(UserFacingError):
        ProjectStore(tmp_path).create(name)


def test_create_rejects_duplicate(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("歌")
    with pytest.raises(UserFacingError, match="已經有"):
        store.create("歌")


def test_state_survives_reload_and_folder_move(tmp_path):
    store = ProjectStore(tmp_path / "a")
    project = store.create("歌")
    project.set_status("s1_env", DONE, seconds=4.2)
    project.set_status("s2_lyrics", RUNNING)
    project.add_warning("s2_lyrics", "第 3 行太長")
    moved = tmp_path / "b" / "搬家後"
    moved.parent.mkdir()
    project.folder.rename(moved)

    reloaded = ProjectStore(tmp_path / "b").load(moved)
    assert reloaded.status("s1_env") == DONE
    assert reloaded.stage("s1_env")["seconds"] == 4.2
    assert reloaded.stage("s2_lyrics")["warnings"] == ["第 3 行太長"]
    assert reloaded.interrupted_stages() == ["s2_lyrics"]
    assert reloaded.first_unfinished() == "s2_lyrics"


def test_mark_stale_only_touches_downstream_stages_that_ran(tmp_path):
    project = ProjectStore(tmp_path).create("歌")
    for stage_id in ("s1_env", "s2_lyrics", "s3_song", "s4_subtitle"):
        project.set_status(stage_id, DONE)
    project.set_status("s5_video", FAILED)

    changed = project.mark_stale_after("s2_lyrics")       # 例如手動改了歌詞

    assert changed == ["s3_song", "s4_subtitle", "s5_video"]
    assert project.status("s1_env") == DONE
    assert project.status("s2_lyrics") == DONE
    assert project.status("s3_song") == STALE
    assert project.status("s6_compose") == PENDING         # 還沒跑過的不標過期
    assert project.first_unfinished() == "s3_song"


def test_subtitle_style_change_only_invalidates_compose_and_export(tmp_path):
    project = ProjectStore(tmp_path).create("歌")
    for stage_id in STAGE_IDS:
        project.set_status(stage_id, DONE)
    assert project.mark_stale_after("s5_video") == ["s6_compose", "s7_export"]


def test_list_skips_folders_that_are_not_projects(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("甲")
    (tmp_path / "不是專案").mkdir()
    (tmp_path / "壞掉").mkdir()
    (tmp_path / "壞掉" / "project.json").write_text("{ 壞掉", encoding="utf-8")
    assert [p.name for p in store.list_projects()] == ["甲"]


def test_corrupt_project_file_gives_readable_error(tmp_path):
    (tmp_path / "project.json").write_text("{", encoding="utf-8")
    with pytest.raises(UserFacingError, match="專案檔損毀"):
        ProjectStore(tmp_path).load(tmp_path)


def test_reset_to_defaults_keeps_files_but_marks_stages_stale(tmp_path):
    project = ProjectStore(tmp_path).create("歌", {"engine": "ltx23"})
    project.data["inputs"].update(content="舊的內容", style="搖滾", photo="input/photo_original.png")
    project.data["mode"] = "step"
    project.data["params"] = {"subtitle_style": {"font": "標楷體"}}
    project.path("input", "photo_original.png").write_bytes(b"x")
    project.path("lyrics", "lyrics.txt").write_text("舊歌詞", encoding="utf-8")
    for stage_id in ("s1_env", "s2_lyrics", "s3_song"):
        project.set_status(stage_id, "done")

    project.reset({"engine": "infinitetalk", "resolution": "480p"}, clear_files=False)

    loaded = ProjectStore(tmp_path).load(project.folder)
    assert loaded.data["inputs"] == {"engine": "infinitetalk", "resolution": "480p", "photo": "input/photo_original.png"}
    assert loaded.data["mode"] == "auto" and loaded.data["params"] == {}
    assert [loaded.status(s) for s in ("s1_env", "s2_lyrics", "s3_song", "s4_subtitle")] == \
        ["done", "stale", "stale", "pending"]
    assert loaded.path("lyrics", "lyrics.txt").exists() and loaded.path("input", "photo_original.png").exists()


def test_reset_clear_all_removes_everything_but_the_project(tmp_path):
    project = ProjectStore(tmp_path).create("歌", {})
    project.data["inputs"].update(content="舊的內容", photo="input/photo_original.png")
    project.data["seeds"] = {"yue2": 1}
    project.data["song_takes"] = [{"index": 1}]
    project.path("input", "photo_original.png").write_bytes(b"x")
    project.path("audio", "takes").mkdir()
    project.path("audio", "takes", "a.flac").write_bytes(b"x")
    (project.folder / "final_mv.mp4").write_bytes(b"x")
    project.set_status("s3_song", "done", seconds=3.0)
    outside = tmp_path / "別的專案.txt"
    outside.write_text("不能動", encoding="utf-8")

    project.reset({"engine": "ltx23"}, clear_files=True)

    loaded = ProjectStore(tmp_path).load(project.folder)
    assert loaded.name == "歌" and loaded.data["inputs"] == {"engine": "ltx23"}
    assert loaded.data["seeds"] == {} and loaded.data["song_takes"] == []
    assert all(loaded.stage(s) == {"status": "pending"} for s in ("s1_env", "s3_song", "s7_export"))
    assert sorted(p.name for p in project.folder.iterdir()) == sorted(["project.json", *SUBDIRS])
    assert not any(any((project.folder / sub).iterdir()) for sub in SUBDIRS)
    assert outside.read_text(encoding="utf-8") == "不能動"
