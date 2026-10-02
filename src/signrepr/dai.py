"""Parse public DAI mapping evidence; never download login-gated media or labels."""
import re
import unicodedata
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup


def normalize_text(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip().casefold()


def candidates_from_html(html, page_url):
    soup = BeautifulSoup(html, 'html.parser')
    candidates = []
    for source in soup.select('td.datasource[id^="ds_"]'):
        index = source['id'][3:]
        utterance = source.get_text(' ', strip=True)
        gloss = soup.find('div', id='fg_' + index)
        frontal = soup.find('div', id='uv_' + index)
        if not gloss or not frontal:
            continue
        text = gloss.get_text(' ', strip=True)
        translation = re.search(r'engl trans:\s*(.*)$', text, re.S)
        videos = [urljoin(page_url, node['src']) for node in frontal.find_all('source', src=True)
                  if urlsplit(node['src']).path.endswith('.mp4')]
        candidates.append({'canonical_utterance_id': utterance,
                           'recording_group': 'asllrp:' + utterance.rsplit('-U-', 1)[0],
                           'source_translation': translation.group(1).strip() if translation else None,
                           'frontal_video_urls': videos, 'source_gloss_and_nmm_text': text,
                           'view_label': frontal.get_text(' ', strip=True),
                           'source_recording_name': utterance.rsplit('-U-', 1)[0]})
    return candidates, 'To download data, you must be logged in' in soup.get_text(' ', strip=True)


def resolve_row(row, candidates):
    matching = [candidate for candidate in candidates
                if candidate['source_translation'] is not None
                and normalize_text(candidate['source_translation']) == normalize_text(row['matched translation'])]
    if len(matching) != 1:
        return {'mapping_status': 'UNRESOLVED', 'reason': 'AMBIGUOUS_TRANSLATION_MATCH' if len(matching) > 1 else 'NO_EXACT_TRANSLATION_MATCH',
                'candidate_count': len(candidates), 'translation_matches': len(matching)}
    chosen = matching[0]
    raw_source = row['video_id'].rsplit('_', 1)[0]
    if raw_source != chosen['source_recording_name']:
        return {'mapping_status': 'UNRESOLVED', 'reason': 'RAW_RECORDING_ID_CONFLICT', 'raw_recording_candidate': raw_source,
                'page_recording_candidate': chosen['source_recording_name']}
    if len(chosen['frontal_video_urls']) != 1 or 'Front' not in chosen['view_label']:
        return {'mapping_status': 'UNRESOLVED', 'reason': 'FRONTAL_VIEW_UNRESOLVED'}
    return {'mapping_status': 'RESOLVED_METADATA_REVIEW_PENDING', **chosen,
            'resolution_evidence': 'Unique exact matched-translation match plus matching raw recording prefix and explicit Front view',
            'linguistic_video_alignment_human_review': 'NOT_PERFORMED',
            'media_download_status': 'NOT_ATTEMPTED', 'scope_gold_status': 'NOT_OBTAINED'}
