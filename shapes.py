from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import plotly.graph_objects as go

class MusicOnTrajectory:
    def __init__(self, df, shape):
        self.df = df
        self.shape = shape

    def run(self):
      closest_songs = self.shape.find_closest_songs(self.df)
      fig = self.shape.plot_closest_points(closest_songs)

      print("Figure created:", fig)
      return fig

class Line:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.m = None
        self.c = None

    def set_slope(self, m):
      self.m = m

    def set_intercept(self, c):
      self.c = c

    def equation(self, x):
        return self.m * x + self.c

    def perpendicular_distance(self, x, y):
        y_line = self.equation(x)
        return np.abs(y - y_line) / np.sqrt(self.m**2 + 1)

    def set_m_c(self, df):
      other_x = df.loc[:, 'valence'].values
      other_y = df.loc[:, 'arousal'].values
      # Calculate the slope of the line passing through the specific point
      # and minimize the error with the other points
      m = np.sum((other_x - self.x) * (other_y - self.y)) / np.sum((other_x - self.x)**2)
      c = self.y - m * self.x
      self.set_slope(m)
      self.set_intercept(c)

    def find_closest_songs(self, df):
        self.set_m_c(df)
        df['distance_to_line'] = self.perpendicular_distance(df['valence'], df['arousal'])
        closest_songs = df.sort_values(by='distance_to_line').head(10)
        return closest_songs

    def plot_closest_points(self, closest_songs):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=closest_songs['valence'],
            y=closest_songs['arousal'],
            mode='markers',
            marker=dict(color=closest_songs['colour']),
            text=closest_songs['track'],
            hoverinfo='text+x+y',  # Display track and artist information on hover
            showlegend=False
        ))

        x_values = np.linspace(min(closest_songs['valence']), max(closest_songs['valence']), 100)
        y_values = self.equation(x_values)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color='black'),
            name='Best-Fit Line'
        ))

        fig.update_layout(
            xaxis_title='Valence',
            yaxis_title='Arousal',
            title='Valence-Arousal Graph'
        )

        print("--- Closest tracks to trajectory ---")
        for i, track in enumerate(closest_songs['track'], start=1):
            print(f"{i}: {track}")

        return fig

class Circle:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.x_circle = None
        self.y_circle = None

    def calculate_radius(self, df, scale = 0.5):
        radius = scale * np.sqrt((df['valence'] - self.x)**2 + (df['arousal'] - self.y)**2).max()
        return radius

    def set_circle_points(self, radius):
        theta = np.linspace(0, 2 * np.pi, 9)
        circle_points_x = self.x + radius * np.cos(theta)
        circle_points_y = self.y + radius * np.sin(theta)

        self.x_circle = circle_points_x
        self.y_circle = circle_points_y

        return circle_points_x, circle_points_y

    def find_closest_songs(self, df):
      radius = self.calculate_radius(df)
      circle_points_x, circle_points_y = self.set_circle_points(radius)
      points = []
      visited_indexes = set()
      for i in range(len(circle_points_x)):
          distances = np.sqrt((df['valence'] - circle_points_x[i])**2 + (df['arousal'] - circle_points_y[i])**2)
          closest_indexes = np.argsort(distances)
          for index in closest_indexes:
              if index not in visited_indexes and index != len(df)-1:
                  points.append(index)
                  visited_indexes.add(index)
                  break
      closest_songs = df.iloc[points]
      return closest_songs

    def plot_closest_points(self, closest_songs):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closest_songs['valence'],
            y=closest_songs['arousal'],
            mode='markers',
            marker=dict(color=closest_songs['colour']),
            text=closest_songs['track'],
            hoverinfo='text+x+y',
            showlegend=False
        ))

        input = df[(df['valence'] == self.x) & (df['arousal'] == self.y)]
        fig.add_trace(go.Scatter(
          x=[self.x],
          y=[self.y],
          mode='markers',
          marker=dict(color=input['colour']),
          text=input['track'],
          hoverinfo='text+x+y',
          showlegend=False
      ))


        fig.add_trace(go.Scatter(x=self.x_circle, y=self.y_circle, mode='lines', line=dict(color='black'), name='Circle'))

        fig.update_layout(
            xaxis_title='Valence',
            yaxis_title='Arousal',
            title='Valence-Arousal Graph')
        
        print("--- Closest tracks to trajectory ---")
        print(f"{1}: {input['track'].values[0]}")
        for i, track in enumerate(closest_songs['track'], start=2):
            print(f"{i}: {track}")

        return fig

class Triangle:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.points = self.generate_triangle_points()

    def generate_triangle_points(self, num_points=3, length=6):
      angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
      return (self.x, self.y) + np.column_stack((length*np.cos(angles), length*np.sin(angles)))

    def find_closest_songs(self, df):
        other_x = df['valence'].values
        other_y = df['arousal'].values

        other_points = np.column_stack((other_x, other_y))

        distances = cdist(other_points, self.points)
        nearest_indexes = np.argsort(distances, axis=0)[:11].flatten()

        nearest_indexes = [d for d in nearest_indexes if not np.array_equal(d, (self.x, self.y))]
        random_indexes = np.random.choice(nearest_indexes, size=9, replace=False)
        closest_songs = df.iloc[random_indexes]
        return closest_songs

    def plot_closest_points(self, closest_songs):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=closest_songs['valence'],
            y=closest_songs['arousal'],
            mode='markers',
            marker=dict(color=closest_songs['colour']),
            text=closest_songs['track'],
            hoverinfo='text+x+y',
            showlegend=False
        ))

        input = df[(df['valence'] == self.x) & (df['arousal'] == self.y)]
        fig.add_trace(go.Scatter(
          x=[self.x],
          y=[self.y],
          mode='markers',
          marker=dict(color=input['colour']),
          text=input['track'],
          hoverinfo='text+x+y',
          showlegend=False
      ))

        fig.add_trace(go.Scatter(
          x=np.append(self.points[:, 0], self.points[0, 0]),
          y=np.append(self.points[:, 1], self.points[0, 1]),
          mode='lines',
          marker=dict(color='black'),
          name='Triangle')
        )

        fig.update_layout(
            xaxis_title='Valence',
            yaxis_title='Arousal',
            title='Valence-Arousal Graph')

        print("--- Closest tracks to trajectory ---")
        print(f"{1}: {input['track'].values[0]}")
        for i, track in enumerate(closest_songs['track'], start=2):
            print(f"{i}: {track}")

        return fig

class Parabola:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.points = self.parabola_points()

    def parabola_points(self, num_points=10, a=1, b=0, c=0, shift=5, scale=0.4):
        x = np.linspace(-5 + shift, 5 + shift, num_points)
        y = (a * (x - shift)**2 + b * (x - shift) + c) * scale
        return np.column_stack((x, y))

    def find_closest_songs(self, df):
        closest_indices = []
        visited_indices = set()

        for i in range(len(self.points)):
            distances = np.sqrt((df['valence'] - self.points[i][0])**2 + (df['arousal'] - self.points[i][1])**2)
            sorted_indices = np.argsort(distances)

            for index in sorted_indices:
                if index not in visited_indices:
                    closest_indices.append(index)
                    visited_indices.add(index)
                    break

        closest_songs = df.iloc[closest_indices]
        return closest_songs

    def plot_closest_points(self, closest_songs):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closest_songs['valence'],
            y=closest_songs['arousal'],
            mode='markers',
            marker=dict(color=closest_songs['colour']),
            text=closest_songs['track'],
            hoverinfo='text+x+y',
            showlegend=False
        ))

        input_point = df[(df['valence'] == self.x) & (df['arousal'] == self.y)]
        fig.add_trace(go.Scatter(
            x=[self.x],
            y=[self.y],
            mode='markers',
            marker=dict(color=input_point['colour']),
            text=input_point['track'],
            hoverinfo='text+x+y',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(x=self.points[:, 0], y=self.points[:, 1], mode='lines', line=dict(color='black'), name='Parabola'))

        fig.update_layout(
            xaxis_title='Valence',
            yaxis_title='Arousal',
            title='Valence-Arousal Graph')

        print("--- Closest tracks to trajectory ---")
        print(f"{1}: {input_point['track'].values[0]}")
        for i, track in enumerate(closest_songs['track'], start=2):
            print(f"{i}: {track}")

        return fig

