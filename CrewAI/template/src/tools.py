from crewai_tools import YoutubeChannelSearchTool
# youtube channel handle to target our search
yt_list = [
    '@IITMadrasBSDegreeProgramme',
    '@krishnaik06',
    '@campusx-official'
]

for yt_channel in yt_list:
    # Initialize the tool with a specific Youtube channel handle
    yt_tool = YoutubeChannelSearchTool(youtube_channel_handle=yt_channel)

    # Search for videos related to 'data science'
    search_results = yt_tool.search_videos(query='data science')

    # Print the search results
    print(f"Search results for {yt_channel}:")
    for video in search_results:
        print(f"- {video['title']} ({video['url']})")
    print("\n")  # Add a newline for better readability between channels