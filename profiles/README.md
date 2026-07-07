# Visual Style Profiles

This directory contains cached visual brand profiles for each client/website.

Profiles are generated automatically by `VisualStyleService` on the first `publish` run
for a client that has Google Drive configured. They are refreshed automatically when:

- The Drive folder changes by more than 5% in image count.
- The profile is older than 30 days.

## Directory structure

```
profiles/
  {client_id}/
    {website_id}/
      visual_style.json
```

## What is stored

Each `visual_style.json` contains:

- `photography_style` — overall photographic style description
- `typical_scenarios` — recurring visual contexts found in the client's photos
- `color_palette` — dominant colors in the brand's photography
- `prompt_guidelines` — direct instructions for DALL-E 3 (appended to every generation prompt)
- `style_description` — comprehensive narrative of the brand's visual identity
- `image_count` — number of images at analysis time (for cache invalidation)
- `analyzed_at` — ISO 8601 timestamp of the last analysis

## To force re-analysis

Delete the `visual_style.json` file for the client. The next `publish` run will
re-analyze the Drive folder and create a fresh profile.

```bash
rm profiles/{client_id}/{website_id}/visual_style.json
```

## Version control

Profiles contain no credentials or sensitive data. You may commit them to version
control to share brand profiles across team members or deployments.

See `profiles/example/example-website/visual_style.json` for a reference format.
