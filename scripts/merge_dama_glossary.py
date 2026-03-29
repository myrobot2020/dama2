#!/usr/bin/env python3
"""Merge glossary sources into ../pali_normalize_map.json (1000+ ASR/Pāli entries)."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "pali_normalize_map.json"
CANON_PATH = Path(__file__).resolve().parent / "dama_canonical_terms.txt"
FLAT_PATH = Path(__file__).resolve().parent / "glossary_flat_pairs.txt"

REMOVE_KEYS = frozenset({"mouse", "force", "perversions"})

# Keys that hijack normal English or are junk (from over-broad earlier merges)
DROP_KEYS = frozenset(
    {
        "ignorance",
        "suffering",
        "no self",
        "hinayana wrong",
        "three baskets",
        "dependent origination",
        "dependent arising",
        "clinging grasping",
        "craving trishna",
        "psychic power",
        "concentration samadhi",
        "consciousness vinnana",
        "formation sankhara",
        "perception sanna",
        "feeling vedana",
        "form matter",
        "morality virtue",
        "wisdom",
        "becoming bhava",
        "birth jati",
        "sorrow shoka",
        "jhana absorption",
        "karuna compassion",
        "mudita sympathetic",
        "upeksha equanimity",
        "shunyata emptiness",
        "dispensation",
        "lamentation",
        "despair",
        "aggregate",
        "theravada school",
        "paali language",
        "araha arhat",
        "samatha vipassanaa",
        "pativedha realization",
        "patipatti practice",
        "pariyatti study",
        "pativedha penetration",
    }
)


def fold_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if ord(c) < 128).lower()


def load_canonical_lines() -> list[str]:
    if not CANON_PATH.is_file():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in CANON_PATH.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_flat_pairs() -> list[tuple[str, str]]:
    if not FLAT_PATH.is_file():
        return []
    pairs: list[tuple[str, str]] = []
    for line in FLAT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        wrong, target = line.split("|", 1)
        wk = wrong.strip().lower()
        tv = target.strip()
        if wk and tv and wk not in DROP_KEYS:
            pairs.append((wk, tv))
    return pairs


# (canonical_target, typo/asr variants) — longest multi-word keys listed in flat file where possible
CORE_GROUPS: list[tuple[str, list[str]]] = [
    ("Tathāgata", ["tata gata", "tata-gata", "tathagatta", "tatagata", "tathagatha"]),
    ("Dhamma", ["dhamm", "dhamaa", "dhumma", "dhammaa", "dharmma", "dharama"]),
    ("Buddha", ["budha", "buddhha", "budhha", "buddah"]),
    ("saṅgha", ["sanga", "sanggha", "sankha", "sangaa", "sanghaa"]),
    ("sutta", ["suttra", "sutah", "sutaa", "suttaa", "sudda"]),
    ("arahant", ["arahath", "arahan", "arahat", "arahantt", "arahanth", "arhat"]),
    ("sotāpanna", ["sotapana", "sothapanna", "sotapan"]),
    ("sakadāgāmī", ["sakadagami", "sakdagami", "sakadagamii"]),
    ("anāgāmī", ["anagami", "anagaami"]),
    ("bhikkhu", ["bikku", "bhikku", "bhikshu", "bikkhuu", "bhikhu"]),
    ("bhikkhunī", ["bhikkhuni", "bikkuni", "bhikshuni"]),
    ("upāsaka", ["upasaka", "upaasaka", "upashaka"]),
    ("upāsikā", ["upasika", "upaasika", "upashika"]),
    ("sāvaka", ["savaka", "savakka", "shawaka"]),
    ("jhāna", ["jhaana", "gyhana", "janaa", "jhanna", "dhyana"]),
    ("samādhi", ["samadi", "samaddhi", "samadhee"]),
    ("vipassanā", ["vipasana", "vipashana", "wipassana", "vipassanna"]),
    ("samatha", ["shamatha", "shamata", "samada"]),
    ("sīla", ["seela", "sheela", "silaa", "sheila"]),
    ("paññā", ["panya", "pannya", "pannaa"]),
    ("mettā", ["mettaa", "metha", "maitri"]),
    ("karuṇā", ["karunaa", "karoona"]),
    ("muditā", ["muditaa"]),
    ("upekkhā", ["upeka", "upekkha", "upekhaa", "upeksha"]),
    ("brahmavihāra", ["brahma vihara", "brahmavihara"]),
    ("ānāpānasati", ["anapana", "anapanasati", "anaapana"]),
    ("satipaṭṭhāna", ["satipatthana", "satipattana", "satipathan", "sati patthana", "satisfacts"]),
    ("bojjhaṅga", ["bojjhanga", "bojangga", "bodjangga"]),
    ("Nibbāna", ["nibanna", "nibbanna", "nivan", "nirvan", "nibban"]),
    ("saṃsāra", ["sangsara", "sansara", "sansaara"]),
    ("kamma", ["kamm", "khamma"]),
    ("taṇhā", ["tanna", "tanhaa", "tahnha"]),
    ("upādāna", ["upaadaana", "upadaan"]),
    ("avijjā", ["avija", "awijja"]),
    ("viññāṇa", ["vignana", "vinnaana", "vinyana"]),
    ("nāmarūpa", ["nama rupa", "namarupa"]),
    ("saḷāyatana", ["salayatana", "saalayatana"]),
    ("vedanā", ["wedana", "vedanna"]),
    ("saṅkhāra", ["sankara", "sangkara", "sankhaara"]),
    ("khandha", ["khanda", "khhandha"]),
    ("anicca", ["anicha", "anichcha", "aniccha", "anitya"]),
    ("dukkha", ["duka", "dhukkha", "dukha", "duhkha"]),
    ("anattā", ["anata", "anattha", "anatman"]),
    ("Paṭiccasamuppāda", ["paticca samuppada", "patichcha", "paticcasamuppada"]),
    ("āsava", ["asawa", "aasava", "ashava"]),
    ("vipallāsa", ["vipallasa", "vipalasa", "vipallas", "palasa"]),
    ("gandhabba", ["gandaba", "gandharba", "gandharv"]),
    ("yakkha", ["yaka", "yakha", "yaksha"]),
    ("brāhmaṇa", ["bramin", "brahmin", "brahmana"]),
    ("Doṇa", ["dhona", "dona", "dhon", "donor"]),
    ("subha", ["suba", "subah", "soobha"]),
    ("ñāṇadassana", ["nyana dasana", "nana dassana"]),
    ("samādhi-bhāvanā", ["samadhi bhavana", "samadi bhaavana"]),
    ("pātimokkha", ["patimokha", "patimokhya", "patimogga"]),
    ("indriya-saṃvara", ["indriya samvara", "indriya sangvara"]),
    ("vijjā-caraṇa", ["vijja carana", "vija charana", "fija charana", "vidya charana"]),
    ("sati-sampajañña", ["sati sampajanna", "sathi sampajanna"]),
    ("Brahmabhūta", ["brahmabhuta", "brahma bhuta"]),
    ("Dhammabhūta", ["dhammabhuta", "dhama bhuta"]),
    ("sammā-diṭṭhi", ["samma ditthi", "sama ditthi"]),
    ("sammā-saṅkappa", ["samma sankappa", "samma sankapa"]),
    ("sammā-vācā", ["samma vaca", "samma vacha"]),
    ("sammā-kammanta", ["samma kamanta"]),
    ("sammā-ājīva", ["samma ajiva", "samma ajeeva"]),
    ("sammā-vāyāma", ["samma vayama", "samma vyayama"]),
    ("sammā-sati", ["sama sati"]),
    ("sammā-samādhi", ["samma samadi"]),
    ("ariya", ["ariyan", "arya"]),
    ("puñña", ["punya", "punny"]),
    ("cetanā", ["chetana"]),
    ("sāsana", ["shasana"]),
    ("pariyatti", ["pariyaratti"]),
    ("paṭipatti", ["patipatti"]),
    ("paṭivedha", ["pativeda"]),
    ("nibbidā", ["nibida", "nibbida"]),
    ("virāga", ["viraga", "veeraga"]),
    ("vimutti", ["vimuthi", "vimukti"]),
    ("amata", ["amatha", "ammata"]),
    ("kāyagatāsati", ["kayagata sati", "kaya gata sati"]),
    ("iddhipāda", ["iddhipada"]),
    ("vitakka", ["vitaka"]),
    ("saññā", ["sannya", "sanna"]),
    ("attā", ["atma"]),
    ("abhidhamma", ["abhi dhamma"]),
    ("vinaya", ["vinay"]),
    ("abhiññā", ["abhinn", "abhinnya", "abhijna"]),
    ("bodhipakkhiya", ["bodhi pakkhiya"]),
    ("kilesa", ["kilesha"]),
    ("nīvaraṇa", ["nivarana", "nivaranna"]),
    ("saṃyojana", ["sanyojana", "samyojana"]),
    ("māra", ["maara"]),
    ("moha", ["mooha"]),
    ("lobha", ["lobh"]),
    ("dosa", ["dosha", "dossa"]),
    ("māna", ["maana"]),
    ("diṭṭhi", ["ditthi", "drishti"]),
    ("micchā", ["miccha", "michcha"]),
    ("sukha", ["sukh", "sukkha"]),
    ("pīti", ["piti", "peeti"]),
    ("passaddhi", ["passadhi", "prasadi"]),
    ("ekaggatā", ["ekaggata", "ekagrata"]),
    ("saddhā", ["saddha", "sraddha", "shraddha"]),
    ("Theravāda", ["theravada", "therawaada"]),
    ("Tipiṭaka", ["tipitaka", "tripitaka"]),
    ("Anguttara Nikaya", ["angutta nikaya", "anguttara nikaaya", "aṅguttara"]),
    ("Majjhima Nikāya", ["majjhima nikaya", "majjima nikaya"]),
    ("Dīgha Nikāya", ["digha nikaya", "deega nikaya"]),
    ("Saṃyutta Nikāya", ["samyutta nikaya", "sanyutta nikaya"]),
    ("Khuddaka Nikāya", ["khuddaka nikaya"]),
    ("Sutta Piṭaka", ["sutta pitak"]),
    ("Vinaya Piṭaka", ["vinaya pitak"]),
    ("Abhidhamma Piṭaka", ["abhidhamma pitaka"]),
    ("Dhammapada", ["dhammapad"]),
    ("paṭisambhidā", ["patisambhida"]),
    ("yathābhūta", ["yathabhuta", "yatha bhuta"]),
    ("samatha-vipassanā", ["samatha vipassana"]),
    ("āsavakkhaya", ["asavakkhaya", "asava khaya"]),
    ("ñāṇa", ["nana", "nyana", "gnana"]),
    ("dassana", ["darshana"]),
    ("jhānaṅga", ["jhananga"]),
    ("rūpa", ["rupa", "roopa"]),
    ("arūpa", ["arupa"]),
    ("deva", ["deeva"]),
    ("sacca", ["satya"]),
    ("samudaya", ["samodaya"]),
    ("nirodha", ["nirhod"]),
    ("magga", ["magg", "marg"]),
    ("paṭipadā", ["patipada", "patipadda"]),
    ("kasiṇa", ["kasina"]),
    ("marananussati", ["marana anussati"]),
    ("buddhānussati", ["buddhanussati"]),
    ("dhammānussati", ["dhammanussati"]),
    ("saṅghānussati", ["sanghanussati"]),
    ("sīlānussati", ["silanussati"]),
    ("cāgānussati", ["caganussati"]),
    ("devatānussati", ["devatanussati"]),
    ("upasamānussati", ["upasamanussati"]),
    ("dibbacakkhu", ["dibba cakkhu", "divya chakshu"]),
    ("dibbasota", ["dibba sota"]),
    ("cetopariyañāṇa", ["cetopariya nana"]),
    ("pubbenivāsānussati", ["pubbenivasanussati"]),
    ("cutūpapātañāṇa", ["cutupapata nana"]),
    ("āsavakkhayañāṇa", ["asavakkhaya nana", "asava khaya nana"]),
    ("parinibbāna", ["parinibbana"]),
    ("kammaṭṭhāna", ["kammatthana", "kamathana"]),
    ("gotrabhu", ["gotrabhoo"]),
    ("pāḷi", ["pali"]),
    ("aṭṭhakathā", ["atthakatha", "arthakatha"]),
    ("ṭīkā", ["tika", "teeka"]),
    ("vicikicchā", ["vicikiccha", "vichikitsa"]),
    ("uddhacca", ["uddhacha", "uddhachya"]),
    ("macchariya", ["machchariya"]),
    ("thīnamiddha", ["thina middha"]),
    ("kukkucca", ["kukucha"]),
    ("vicāra", ["vichara"]),
    ("suññatā", ["sunnata"]),
    ("animitto", ["animitta"]),
    ("appāṇihita", ["appanihita"]),
    ("byākaraṇa", ["byakarana"]),
    ("abyākata", ["abyakata"]),
    ("pañca", ["pancha"]),
    ("kāya", ["kaya", "kaaya"]),
    ("citta", ["chitta"]),
    ("cetasika", ["chetasika"]),
    ("phassa", ["sparsha"]),
    ("bhava", ["bhawa"]),
    ("jāti", ["jaati"]),
    ("jarāmaraṇa", ["jara marana", "jaramarana"]),
    ("paṭiccasamuppāda", ["paticcasamuppada"]),
    ("paṭisandhi", ["patisandhi"]),
    ("cuti", ["chuti"]),
    ("ānāpāna", ["anapana"]),
    ("adukkhamasukhā", ["adukkhamasukha", "a duhkha ma sukha"]),
    ("ākāsa", ["akasa", "akasha"]),
    ("nevasaññānāsaññā", ["nevasanna nasanna"]),
    ("nirodhasamāpatti", ["nirodha samapatti"]),
    ("saññāvedayitanirodha", ["sannavedayitanirodha"]),
    ("iddhi", ["riddhi"]),
    ("bodhisatta", ["bodhisattva", "bodhisatva"]),
    ("paccekabuddha", ["pacceka buddha"]),
    ("sammāsambuddha", ["sammasambuddha"]),
    ("mahāsāvaka", ["maha savaka"]),
    ("ācariya", ["acariya", "acharya"]),
    ("upajjhāya", ["upajjhaya"]),
    ("paṭikkūlamanasikāra", ["patikkulamanasikara"]),
    ("asubha", ["ashubh"]),
    ("āghātapaṭisaṃyutta", ["aghatapatisamyutta"]),
    ("saddhammapatirūpaka", ["saddhamma patirupaka"]),
    ("saddhamma", ["sad dhamma"]),
    ("dhammacakka", ["dhamma cakka"]),
    ("sotāpatti", ["sotapatti", "sotapati"]),
    ("sakadāgāmitā", ["sakadagamita"]),
    ("anāgāmitā", ["anagamita"]),
    ("vodāna", ["vodaana"]),
    ("paññindriya", ["pannindriya"]),
    ("samādhindriya", ["samadhindriya"]),
    ("paññābala", ["pannabala"]),
    ("saddhābala", ["saddhabala"]),
    ("samādhibala", ["samadhibala"]),
    ("ottappabala", ["ottappabal"]),
    ("abyāpāda", ["abyapada", "abyapajj"]),
    ("abyāpanna", ["abyapann"]),
    ("micchā-diṭṭhi", ["micchaditthi"]),
    ("micchā-saṅkappa", ["micchasankappa"]),
    ("micchā-vācā", ["micchavaca"]),
    ("micchā-kammanta", ["micchakammanta"]),
    ("micchā-ājīva", ["micchaajiva"]),
    ("micchā-vāyāma", ["micchavayama"]),
    ("micchā-sati", ["micchasati"]),
    ("micchā-samādhi", ["micchasamadhi"]),
    ("vipākadhamma", ["vipakadhamma", "vipakadhamm"]),
    ("vipākadhammābhisankhāra", ["vipakadhammabhisankhara", "vipakadhammabhisankhar"]),
    ("vipākadhammābhisankhata", ["vipakadhammabhisankhata", "vipakadhammabhisankhat"]),
    ("abyākatavipāka", ["abyakatavipaka"]),
    ("abyākatahetuka", ["abyakatahetuka"]),
    ("kusalahetuka", ["kusalahetuka"]),
    ("akusalahetuka", ["akusalahetuka"]),
    ("abyākatahetu", ["abyakatahetu"]),
    ("abyākatamūla", ["abyakatamula"]),
    ("akusalamūla", ["akusalamula"]),
    ("kusalamūla", ["kusalamula"]),
    ("dasa kusala kammapath", ["dasakusalakammapath"]),
    ("dasa kusala", ["dasakusal"]),
    ("akusala kammapath", ["akusalakammapath"]),
    ("abyākata kammapath", ["abyakatakammapath"]),
    ("dasa kammapath", ["dasakammapath"]),
    ("kammapath", ["kammapath"]),
    ("kusala vipāka", ["kusala vipaka"]),
    ("akusala vipāka", ["akusala vipaka"]),
    ("abyākata vipāka", ["abyakata vipaka"]),
    ("mettācitta", ["mettacitt"]),
    ("karuṇācitta", ["karunacitt"]),
    ("muditācitta", ["muditacitt"]),
    ("upekkhācitta", ["upekkacitt"]),
    ("sammappadhāna", ["sammappadhana"]),
    ("ekodibhāva", ["ekodibhava"]),
    ("paṭisotagāmī", ["patisotagami"]),
    ("anulomika khanti", ["anulomikakhanti"]),
    ("paṭilomika khanti", ["patilomikakhanti"]),
    ("paṭipannaka bhikkhu", ["patipannakabhikkhu"]),
    ("paṭipannaka bhikkhunī", ["patipannakabhikkhuni"]),
    ("paṭipannaka saṅgha", ["patipannakasangha"]),
    ("paṭipannaka sāvaka", ["patipannakasavaka"]),
    ("paṭipannaka upāsaka", ["patipannakaupasaka"]),
    ("paṭipannaka upāsikā", ["patipannakaupasika", "patipannakaupasik"]),
]


def merge_glossary() -> dict[str, str]:
    raw = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("pali_normalize_map.json must be a JSON object")

    merged: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        kl = k.strip()
        if kl.lower() in REMOVE_KEYS:
            continue
        if kl.lower() in DROP_KEYS:
            continue
        merged[kl] = v.strip()

    for line in load_canonical_lines():
        key = fold_ascii(line)
        if not key or key == line.lower():
            continue
        if key in DROP_KEYS:
            continue
        merged.setdefault(key, line)

    for target, wrongs in CORE_GROUPS:
        t = target.strip()
        if not t:
            continue
        for w in wrongs:
            wk = w.strip().lower()
            if len(wk) < 2 or wk in DROP_KEYS:
                continue
            merged.setdefault(wk, t)

    for wk, tv in load_flat_pairs():
        if wk in DROP_KEYS:
            continue
        merged.setdefault(wk, tv)

    merged = {k: v for k, v in merged.items() if k.lower() not in DROP_KEYS}
    return merged


def main() -> None:
    merged = merge_glossary()
    items = sorted(merged.items(), key=lambda kv: (-len(kv[0]), kv[0].lower()))
    out = {k: v for k, v in items}
    MAP_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} entries to {MAP_PATH}")


if __name__ == "__main__":
    main()
