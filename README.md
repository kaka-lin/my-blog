# My blog

This repo is about my blog that builds with `Hugo`.

## Local development

### Clone repo

1. clone with submodule

```bash
$ git clone --recursive git@github.com:kaka-lin/my-blog-hugo.git
```

2. update after clone

```bash
$ git submodule update --init --recursive
```

> Update the submodule to the latest remote commit, as below:
> ```
> $ git submodule update --remote --merge
> ```

### Draft

```bash
$ hugo server -D
```

### Deploy

```bash
$ ./deploy.sh
```
