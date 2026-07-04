# immich scripts

this is a repo of scripts that use the immich api to turn albums into containers for [photos.samfelton.com](photos.samfelton.com).

## rate.py

uses axif:rating to add 5 star metadata to images in an album based on the output of a NIMA model.

```bash
 python3 rate.py ef33092b-6cf3-440a-b6ea-5942bfc56442 --dry-run  
```

## stack.py

Inspect images in an album and stack them based on similarity. uses axif:rating for deciding primary image.

```bash
python3 stack.py --album ef33092b-6cf3-440a-b6ea-5942bfc56442 --strictness 0.8
```