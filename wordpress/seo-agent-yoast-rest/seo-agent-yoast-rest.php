<?php
/**
 * Plugin Name:  SEO Agent — Yoast REST Fields
 * Plugin URI:   https://github.com/your-org/seo-agent
 * Description:  Registers Yoast SEO post meta fields with show_in_rest: true so the
 *               SEO Agent publishing pipeline can write them via the WordPress REST API.
 *               Install alongside "SEO Agent Integration" — this plugin only adds the
 *               Yoast field registrations; it does not duplicate any other logic.
 * Version:      1.0.0
 * Requires at least: 5.0
 * Requires PHP: 7.4
 * Author:       SEO Agent
 * License:      MIT
 *
 * Why this plugin is needed
 * ─────────────────────────
 * Yoast SEO (v14+) exposes its data via the yoast/v1 REST namespace for reading,
 * but does NOT register the underlying _yoast_wpseo_* post meta fields with
 * show_in_rest: true in the standard WP REST meta schema.
 *
 * Consequence: a REST API write to those fields is silently discarded by WordPress
 * even when Yoast is active — the HTTP response is 200 OK but the values are never
 * persisted to wp_postmeta.
 *
 * This plugin registers each field through register_post_meta() so that:
 *   1. WordPress persists the values in wp_postmeta (standard REST meta pipeline).
 *   2. Yoast reads them from wp_postmeta via get_post_meta() on the front end.
 *
 * No conflict with Yoast SEO: Yoast checks for existing registrations before
 * registering its own. Even if it does not, WordPress will use the first
 * registration with show_in_rest: true and ignore duplicates.
 *
 * No conflict with "SEO Agent Integration" (seo-agent.php): that plugin only
 * registers the four _seo_agent_* tracking fields. This plugin only registers
 * the ten _yoast_wpseo_* SEO fields. Zero overlap.
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

add_action( 'init', 'seo_agent_yoast_rest_register_meta' );

function seo_agent_yoast_rest_register_meta(): void {

    $auth = static function (): bool {
        return current_user_can( 'edit_posts' );
    };

    $base = [
        'object_subtype' => 'post',   // posts only, not pages or custom types
        'show_in_rest'   => true,
        'type'           => 'string',
        'single'         => true,
        'default'        => '',
        'auth_callback'  => $auth,
    ];

    $fields = [
        '_yoast_wpseo_title' => [
            'description' => 'Yoast SEO — title tag displayed in search results.',
        ],
        '_yoast_wpseo_metadesc' => [
            'description' => 'Yoast SEO — meta description displayed in search results.',
        ],
        '_yoast_wpseo_focuskw' => [
            'description' => 'Yoast SEO — primary focus keyphrase for on-page optimization.',
        ],
        '_yoast_wpseo_canonical' => [
            'description' => 'Yoast SEO — canonical URL to avoid duplicate content issues.',
        ],
        '_yoast_wpseo_opengraph-title' => [
            'description' => 'Yoast SEO — Open Graph title for Facebook and LinkedIn sharing.',
        ],
        '_yoast_wpseo_opengraph-description' => [
            'description' => 'Yoast SEO — Open Graph description for Facebook and LinkedIn sharing.',
        ],
        '_yoast_wpseo_opengraph-image' => [
            'description' => 'Yoast SEO — Open Graph image URL for social sharing previews.',
        ],
        '_yoast_wpseo_twitter-title' => [
            'description' => 'Yoast SEO — Twitter Card title.',
        ],
        '_yoast_wpseo_twitter-description' => [
            'description' => 'Yoast SEO — Twitter Card description.',
        ],
        '_yoast_wpseo_twitter-image' => [
            'description' => 'Yoast SEO — Twitter Card image URL.',
        ],
    ];

    foreach ( $fields as $key => $extra ) {
        register_post_meta( 'post', $key, array_merge( $base, $extra ) );
    }
}
