-- Read-only snapshot of every live series: identity, titles, credits, covers.
SELECT json_agg(x ORDER BY x.title) FROM (
  SELECT s.id, s.title, s.author, s.source_id, s.source, s.cover_book_id, s.library_id,
    (SELECT t.external_id FROM series_trackers t WHERE t.series_id = s.id AND t.provider = 'anilist'
       AND (t.linked_by IS NOT NULL OR t.checked_at IS NOT NULL) LIMIT 1) AS anilist_id,
    o.title AS override_title, o.cover AS override_cover, o.author AS override_author, o.banner AS override_banner,
    a.cover AS art_cover, a.banner AS art_banner,
    d.cover AS auto_cover, d.cover_from AS auto_cover_from, d.author AS auto_author,
    ot.english, ot.romaji, ot.native,
    (SELECT json_agg(DISTINCT al.title) FROM series_alt_titles al WHERE al.series_id = s.id AND al.removed_at IS NULL) AS alt_titles,
    (SELECT json_agg(json_build_array(ss.source_id, ss.source_series_id)) FROM series_sources ss WHERE ss.series_id = s.id) AS sources,
    ta.mode AS art_mode, ta.auto_file, ta.auto_attribution, ta.lookup_status, ta.custom_file
  FROM lib_series s
  LEFT JOIN series_overrides o ON o.series_id = s.id
  LEFT JOIN series_art a ON a.series_id = s.id
  LEFT JOIN series_auto_details d ON d.series_id = s.id
  LEFT JOIN series_other_titles ot ON ot.series_id = s.id
  LEFT JOIN series_title_art ta ON ta.series_id = s.id
  WHERE s.deleted_at IS NULL AND s.merged_into IS NULL
) x;
