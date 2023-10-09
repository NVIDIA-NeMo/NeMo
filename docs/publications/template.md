---
title: Template 2023
author: [First Name, Second Name]
author_gh_user: [Github User 1, Github User 2]
readtime: Time to read in minutes (30)
date: Date of publishing in YYYY-MM-DD format

# Optional: Redirect to a different URL than the blog post page for "Continue reading" link
# Comment out below if you want to show a blog post for the article.
continue_url: "Article URL"

# Categories - Domain of the article
categories: "List categories that this blog post belongs to. They will be displayed on the website."

# Optional: OpenGraph metadata
# og_title: Title of the blog post for Rich URL previews
# og_image: Image for Rich URL previews (absolute URL)
# og_image_type: Image type (e.g. image/png). Defaults to image/png.
# page_path: Relative path to the image from the website root (e.g. /assets/images/). If specified, the image at this path will be used for the link preview. It is unlikely you will need this parameter - you can probably use og_image instead.
# description: Description of the post for Rich URL previews
---

# Notes

- Add folder inside `blogs/posts` directory with year
- Copy the contents of this template to the folder.
- Edit the contents.
- Add a `<!-- more -->` tag to the blog post to indicate where it should say 'Continue reading' in the blogpost preview.
- Send PR and merge.
- ASSETS: All blog images and external content must be hosted somewhere else. Do NOT add things to GitHub for blog contents!

## Note about `continue_url`
    
If "continue_url" is set, AND a URL is specified in "# [Article Title](Article URL)", then there will be no links to the full blog post on the website, although the full blog post will be accessible if a user knows the URL to look for. So the content under the 'more' tag won't be findable by clicking, but will be findable if someone looks for the exact blog post link.

# [Article Title](Article URL)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras in massa et lacus consectetur maximus. Donec fringilla, justo vitae condimentum feugiat, est sapien interdum purus, vel rutrum neque ex quis ipsum. Etiam in mauris odio. 

<!-- more -->

Mauris in mattis massa. Vivamus tempor libero eu ante aliquet tempor. Vestibulum porttitor odio eu ante posuere, sit amet sagittis quam auctor. Nunc sem sem, ultrices eget porta ac, vulputate non nibh. In tempor risus non felis porta, id scelerisque eros interdum.