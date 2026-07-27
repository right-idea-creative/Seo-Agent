<?php
/**
 * wp_compare_server.php — Drop into WordPress root, open in browser.
 * Compare two posts: wp_posts + wp_postmeta + taxonomies.
 * DELETE THIS FILE after use.
 *
 * Usage: https://overheaddoorgnv.com/wp_compare_server.php?a=8056&b=8044&token=SECRET
 */

// ── CHANGE THIS TOKEN before uploading ────────────────────────────────────────
define('SECRET_TOKEN', 'compare_8056_vs_8044_deleteme');

if (empty($_GET['token']) || $_GET['token'] !== SECRET_TOKEN) {
    http_response_code(403);
    die('Forbidden. Add ?token=SECRET_TOKEN to the URL.');
}

$post_a = (int) ($_GET['a'] ?? 0);
$post_b = (int) ($_GET['b'] ?? 0);

if (!$post_a || !$post_b) {
    die('Usage: ?a=POST_ID&b=POST_ID&token=TOKEN');
}

// Bootstrap WordPress
$wp_root = dirname(__FILE__);
define('ABSPATH', $wp_root . '/');
require_once $wp_root . '/wp-load.php';

// ── Output helpers ─────────────────────────────────────────────────────────────
header('Content-Type: text/html; charset=utf-8');
?><!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WP Post Comparison: #<?= $post_a ?> vs #<?= $post_b ?></title>
<style>
body{font-family:monospace;font-size:13px;margin:20px;background:#1e1e1e;color:#d4d4d4}
h2{color:#569cd6;border-bottom:1px solid #555;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin-bottom:24px}
td,th{border:1px solid #444;padding:5px 8px;vertical-align:top;word-break:break-all}
th{background:#2d2d2d;color:#9cdcfe;text-align:left}
.ok{color:#4ec9b0}
.diff{color:#f44747;font-weight:bold}
.only-a{color:#dcdcaa}
.only-b{color:#ce9178}
.key{color:#9cdcfe}
.val{color:#ce9178}
.note{color:#608b4e;font-style:italic}
.section{background:#252526;padding:8px 12px;margin:16px 0 4px;border-left:4px solid #569cd6;font-weight:bold;color:#569cd6}
</style>
</head>
<body>
<?php

global $wpdb;

$a = $post_a;
$b = $post_b;

// ── Fetch wp_posts rows ────────────────────────────────────────────────────────
$row_a = $wpdb->get_row($wpdb->prepare("SELECT * FROM {$wpdb->posts} WHERE ID = %d", $a), ARRAY_A);
$row_b = $wpdb->get_row($wpdb->prepare("SELECT * FROM {$wpdb->posts} WHERE ID = %d", $b), ARRAY_A);

if (!$row_a) die("<p style='color:red'>Post #$a not found.</p>");
if (!$row_b) die("<p style='color:red'>Post #$b not found.</p>");

$skip = ['ID','post_date','post_date_gmt','post_modified','post_modified_gmt',
         'guid','post_name','post_title','post_content','post_excerpt'];

echo "<h2>Post #$a (manual) vs Post #$b (agent)</h2>";

// ── Section 1: wp_posts ────────────────────────────────────────────────────────
echo '<div class="section">[1/3] wp_posts columns</div>';
echo '<table><tr><th>column</th><th>#'.$a.' (manual)</th><th>#'.$b.' (agent)</th><th>status</th></tr>';

$post_diffs = [];
foreach ($row_a as $col => $val_a) {
    if (in_array($col, $skip)) continue;
    $val_b = $row_b[$col] ?? '(missing)';
    $same  = ($val_a === $val_b);
    $cls   = $same ? 'ok' : 'diff';
    $mark  = $same ? '✓' : '✗';
    if (!$same) $post_diffs[] = $col;
    echo "<tr class='$cls'><td class='key'>$col</td><td>" . htmlspecialchars((string)$val_a) .
         "</td><td>" . htmlspecialchars((string)$val_b) . "</td><td>$mark</td></tr>";
}
echo '</table>';

$count = count($post_diffs);
echo "<p class='note'>wp_posts: $count column(s) differ" . ($count ? ': ' . implode(', ', $post_diffs) : '') . "</p>";

// ── Section 2: wp_postmeta ─────────────────────────────────────────────────────
echo '<div class="section">[2/3] wp_postmeta — ALL rows (including private Elementor meta)</div>';

$meta_a_rows = $wpdb->get_results($wpdb->prepare(
    "SELECT meta_key, meta_value FROM {$wpdb->postmeta} WHERE post_id = %d ORDER BY meta_key", $a
), ARRAY_A);
$meta_b_rows = $wpdb->get_results($wpdb->prepare(
    "SELECT meta_key, meta_value FROM {$wpdb->postmeta} WHERE post_id = %d ORDER BY meta_key", $b
), ARRAY_A);

$meta_a = [];
foreach ($meta_a_rows as $r) $meta_a[$r['meta_key']] = $r['meta_value'];

$meta_b = [];
foreach ($meta_b_rows as $r) $meta_b[$r['meta_key']] = $r['meta_value'];

$all_keys = array_unique(array_merge(array_keys($meta_a), array_keys($meta_b)));
sort($all_keys);

echo '<table><tr><th>meta_key</th><th>#'.$a.' (manual)</th><th>#'.$b.' (agent)</th><th>status</th></tr>';

$meta_diffs = $meta_only_a = $meta_only_b = [];

foreach ($all_keys as $k) {
    $in_a = isset($meta_a[$k]);
    $in_b = isset($meta_b[$k]);

    if ($in_a && !$in_b) {
        $meta_only_a[] = $k;
        $trunc = mb_strlen($meta_a[$k]) > 150 ? mb_substr($meta_a[$k], 0, 150) . '…' : $meta_a[$k];
        echo "<tr class='only-a'><td class='key'>$k</td><td>" . htmlspecialchars($trunc) .
             "</td><td style='color:#555'>(missing)</td><td>△ only manual</td></tr>";
    } elseif (!$in_a && $in_b) {
        $meta_only_b[] = $k;
        $trunc = mb_strlen($meta_b[$k]) > 150 ? mb_substr($meta_b[$k], 0, 150) . '…' : $meta_b[$k];
        echo "<tr class='only-b'><td class='key'>$k</td><td style='color:#555'>(missing)</td><td>" .
             htmlspecialchars($trunc) . "</td><td>▽ only agent</td></tr>";
    } else {
        $same = ($meta_a[$k] === $meta_b[$k]);
        if (!$same) $meta_diffs[] = $k;
        $cls  = $same ? 'ok' : 'diff';
        $mark = $same ? '✓' : '✗';
        $ta   = mb_strlen($meta_a[$k]) > 150 ? mb_substr($meta_a[$k], 0, 150) . '…' : $meta_a[$k];
        $tb   = mb_strlen($meta_b[$k]) > 150 ? mb_substr($meta_b[$k], 0, 150) . '…' : $meta_b[$k];
        echo "<tr class='$cls'><td class='key'>$k</td><td>" . htmlspecialchars($ta) .
             "</td><td>" . htmlspecialchars($tb) . "</td><td>$mark</td></tr>";
    }
}
echo '</table>';

$d1 = count($meta_diffs); $d2 = count($meta_only_a); $d3 = count($meta_only_b);
echo "<p class='note'>wp_postmeta: $d1 differ, $d2 only-in-manual, $d3 only-in-agent</p>";

// ── Section 3: Taxonomies ──────────────────────────────────────────────────────
echo '<div class="section">[3/3] Taxonomies (wp_term_relationships + wp_term_taxonomy + wp_terms)</div>';

function get_post_terms_full($wpdb, $post_id) {
    return $wpdb->get_results($wpdb->prepare("
        SELECT tt.taxonomy, t.term_id, t.name, t.slug
        FROM {$wpdb->term_relationships} tr
        JOIN {$wpdb->term_taxonomy} tt ON tr.term_taxonomy_id = tt.term_taxonomy_id
        JOIN {$wpdb->terms} t ON tt.term_id = t.term_id
        WHERE tr.object_id = %d
        ORDER BY tt.taxonomy, t.slug
    ", $post_id), ARRAY_A);
}

$terms_a_rows = get_post_terms_full($wpdb, $a);
$terms_b_rows = get_post_terms_full($wpdb, $b);

// Group by taxonomy
$terms_a = $terms_b = [];
foreach ($terms_a_rows as $r) $terms_a[$r['taxonomy']][] = $r['slug'] . ' (#' . $r['term_id'] . ')';
foreach ($terms_b_rows as $r) $terms_b[$r['taxonomy']][] = $r['slug'] . ' (#' . $r['term_id'] . ')';

$all_tax = array_unique(array_merge(array_keys($terms_a), array_keys($terms_b)));
sort($all_tax);

echo '<table><tr><th>taxonomy</th><th>#'.$a.' (manual)</th><th>#'.$b.' (agent)</th><th>status</th></tr>';

$tax_diffs = [];
foreach ($all_tax as $tax) {
    $ta = $terms_a[$tax] ?? [];
    $tb = $terms_b[$tax] ?? [];
    sort($ta); sort($tb);
    $same = ($ta === $tb);
    if (!$same) $tax_diffs[] = $tax;
    $cls  = $same ? 'ok' : 'diff';
    $mark = $same ? '✓' : '✗';
    echo "<tr class='$cls'><td class='key'>$tax</td>" .
         "<td>" . htmlspecialchars(implode(', ', $ta) ?: '(none)') . "</td>" .
         "<td>" . htmlspecialchars(implode(', ', $tb) ?: '(none)') . "</td>" .
         "<td>$mark</td></tr>";
}
echo '</table>';

$nt = count($tax_diffs);
echo "<p class='note'>Taxonomies: $nt differ" . ($nt ? ': ' . implode(', ', $tax_diffs) : '') . "</p>";

// ── Summary ────────────────────────────────────────────────────────────────────
echo '<div class="section">SUMMARY</div>';
echo '<table><tr><th>section</th><th>metric</th><th>count</th></tr>';
echo "<tr><td>wp_posts</td><td>columns differ</td><td>" . count($post_diffs) . "</td></tr>";
echo "<tr><td>wp_postmeta</td><td>keys differ</td><td>$d1</td></tr>";
echo "<tr><td>wp_postmeta</td><td>only in manual (#$a)</td><td>$d2</td></tr>";
echo "<tr><td>wp_postmeta</td><td>only in agent (#$b)</td><td>$d3</td></tr>";
echo "<tr><td>taxonomies</td><td>differ</td><td>$nt</td></tr>";
echo '</table>';

echo '<p class="note">DELETE this file from the server after use.</p>';
echo '</body></html>';
