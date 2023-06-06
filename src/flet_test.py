import flet as ft

import flet
from flet import (
    Control,
    Column,
    Container,
    IconButton,
    Page,
    Row,
    Icon,
    Text,
    IconButton,
    colors,
    icons,
    AppBar,
    PopupMenuButton,
    PopupMenuItem,
    margin,
    FletApp,
)
import hvplot.streamz

from streamz.dataframe import Random

df = Random(interval="200ms", freq="50ms")


def get_card(i):
    return ft.Card(
        content=ft.Container(
            content=ft.Column(
                [
                    FletApp(
                        url="http://localhost:5006/holoviz_app", width=400, height=400
                    ),
                    # ft.Image(
                    #     src=f"https://picsum.photos/150/150?{i}",
                    #     fit=ft.ImageFit.FILL,
                    #     height=400,
                    #     width=600,
                    #     repeat=ft.ImageRepeat.NO_REPEAT,
                    #     border_radius=ft.border_radius.all(10),
                    # ),
                    # ft.Row(
                    #     [ft.TextButton("Buy tickets"), ft.TextButton("Listen")],
                    #     alignment=ft.alignment.bottom_right,
                    # ),
                ],
                alignment=ft.alignment.bottom_right,
            ),
            width=600,
            height=800,
            padding=100,
        )
    )


def create_cards(images, max_num=10):
    for i in range(0, max_num):
        images.controls.append(get_card(i))


def main(page: ft.Page):
    page.title = "GridView Example"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 50
    page.update()

    appbar_items = [
        PopupMenuItem(text="Login"),
        PopupMenuItem(),  # divider
        PopupMenuItem(text="Settings"),
    ]
    page.appbar = AppBar(
        leading_width=100,
        title=Text("Imitation Learning", size=32, text_align="start"),
        center_title=False,
        toolbar_height=75,
        bgcolor=colors.LIGHT_BLUE_ACCENT_700,
        actions=[
            Container(
                content=PopupMenuButton(items=appbar_items),
                margin=margin.only(left=50, right=25),
            )
        ],
    )

    images = ft.GridView(
        expand=4,
        runs_count=2,
        max_extent=800,
        child_aspect_ratio=1.0,
        spacing=5,
        run_spacing=5,
    )
    create_cards(images, max_num=5)

    ## List view for recordings
    lv = ft.ListView(expand=1, spacing=10, padding=20, auto_scroll=True)

    for i in range(0, 10):
        lv.controls.append(ft.Text(f"Recording {i}", size=20))

    record_button = ft.ElevatedButton(text="Record")
    save_button = ft.ElevatedButton(text="Save button")

    sidebar = ft.Column([record_button, save_button, lv], expand=1, width=100)

    row = ft.Row([sidebar, images], spacing=10, run_spacing=10, expand=1)

    page.add(row)

    page.update()


ft.app(target=main, assets_dir="assets")
