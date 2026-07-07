<?php
/**
 * Plugin Name: SEO Agent Integration
 * Plugin URI:  https://github.com/your-org/seo-agent
 * Description: Registers post meta fields for the SEO-Agent publishing pipeline.
 *              Required for idempotent publishing, audit traceability, and Yoast SEO
 *              metadata writes via the WordPress REST API.
 * Version:     1.1.0
 * Requires at least: 5.0
 * Requires PHP: 7.4
 *
 * Installation:
 *   Copy this directory to wp-content/plugins/seo-agent/ and activate the plugin
 *   from the WordPress admin panel (Plugins → Activate).
 *
 * What this plugin does:
 *   1. Registers four custom post meta fields (_seo_agent_*) for idempotent publishing.
 *   2. Registers all Yoast SEO post meta fields with show_in_rest: true so the
 *      SEO-Agent tool can write them via the WordPress REST API.
 *
 *   Why Yoast fields must be registered here:
 *   Yoast SEO (v14+) exposes its data via the yoast/v1 REST namespace for reading,
 *   but does NOT register the underlying _yoast_wpseo_* post meta fields with
 *   show_in_rest: true in the standard WP REST meta schema. This means that a REST
 *   API write to those fields is silently discarded by WordPress even when Yoast is
 *   active. By registering them here, we allow WordPress to persist the values in
 *   wp_postmeta, where Yoast reads them via get_post_meta().
 */

if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

add_action( 'init', 'seo_agent_register_meta' );

function seo_agent_register_meta(): void {

    $auth = function () {
        return current_user_can( 'edit_posts' );
    };

    $base = [
        'show_in_rest'  => true,
        'type'          => 'string',
        'single'        => true,
        'default'       => '',
        'auth_callback' => $auth,
    ];

    // ── SEO Agent tracking fields ─────────────────────────────────────────────

    $agent_fields = [
        '_seo_agent_id' => [
            'description' => 'Unique article UUID assigned by SEO-Agent. Used for idempotent publishing.',
        ],
        '_seo_agent_schema_version' => [
            'description' => 'Article data-model version at time of publish (e.g. "1.0").',
        ],
        '_seo_agent_prompt_version' => [
            'description' => 'Prompt version used to generate this article (e.g. "1.0").',
        ],
        '_seo_agent_model' => [
            'description' => 'Claude model used to generate this article (e.g. "claude-opus-4-8").',
        ],
    ];

    foreach ( $agent_fields as $key => $extra ) {
        register_post_meta( 'post', $key, array_merge( $base, $extra ) );
    }

    // ── Yoast SEO fields ──────────────────────────────────────────────────────
    //
    // Yoast does not register these with show_in_rest in modern versions.
    // We register them here so the REST API persists them in wp_postmeta,
    // which is exactly where Yoast reads them on the front end.

    $yoast_fields = [
        '_yoast_wpseo_title'                => 'Yoast SEO title tag.',
        '_yoast_wpseo_metadesc'             => 'Yoast meta description.',
        '_yoast_wpseo_focuskw'              => 'Yoast focus keyphrase.',
        '_yoast_wpseo_canonical'            => 'Yoast canonical URL.',
        '_yoast_wpseo_opengraph-title'      => 'Yoast Open Graph title.',
        '_yoast_wpseo_opengraph-description'=> 'Yoast Open Graph description.',
        '_yoast_wpseo_opengraph-image'      => 'Yoast Open Graph image URL.',
        '_yoast_wpseo_twitter-title'        => 'Yoast Twitter card title.',
        '_yoast_wpseo_twitter-description'  => 'Yoast Twitter card description.',
        '_yoast_wpseo_twitter-image'        => 'Yoast Twitter card image URL.',
    ];

    foreach ( $yoast_fields as $key => $description ) {
        register_post_meta( 'post', $key, array_merge( $base, [
            'description' => $description,
        ] ) );
    }
}
