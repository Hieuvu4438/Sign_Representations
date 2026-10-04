"""Explicit release scheme-less URL extension; original audit remains frozen."""
from msasl_metadata import youtube_id as original_id, issues as original_issues


def youtube_id(url):
    if isinstance(url,str) and url.startswith('www.youtube.com/watch?'):
        return original_id('https://'+url)
    return original_id(url)


def issues(row,class_count):
    result=original_issues(row,class_count)
    if isinstance(row,dict) and youtube_id(row.get('url')) is not None:
        result=[issue for issue in result if issue!='source_url_not_accepted']
    return result
