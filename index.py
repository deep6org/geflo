import click


@click.group()
def geflo():
    """A CLI wrapper for the NFT purchasing of electronic quantum states files."""


if __name__ == '__main__':
    geflo(prog_name='geflo')
    