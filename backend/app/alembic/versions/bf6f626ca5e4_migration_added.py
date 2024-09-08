"""migration added

Revision ID: bf6f626ca5e4
Revises: 36d7ec60e107
Create Date: 2024-08-03 23:31:18.423282

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'bf6f626ca5e4'
down_revision = '36d7ec60e107'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('lookup_jobtitles',
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('jobtitle', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('lookup_jobtype',
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('jobtype', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user_onboarding',
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('user_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('urgency_level', sa.Integer(), nullable=False),
    sa.Column('h1b_bool', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user_resumes',
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('user_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('blob_uri', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('uploaded', sa.Boolean(), nullable=False),
    sa.Column('uploaded_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('pref_jobs',
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('location', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('location_coordinates', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('remote_bool', sa.Boolean(), nullable=False),
    sa.Column('h1b_bool', sa.Boolean(), nullable=False),
    sa.Column('onboarding_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.ForeignKeyConstraint(['onboarding_id'], ['user_onboarding.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('pref_jobtitle',
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('jobtitle_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('pref_jobs_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('user_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.ForeignKeyConstraint(['jobtitle_id'], ['lookup_jobtitles.id'], ),
    sa.ForeignKeyConstraint(['pref_jobs_id'], ['pref_jobs.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('pref_jobtype',
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('jobtype_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('pref_jobs_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('user_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.ForeignKeyConstraint(['jobtype_id'], ['lookup_jobtype.id'], ),
    sa.ForeignKeyConstraint(['pref_jobs_id'], ['pref_jobs.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('pref_jobtype')
    op.drop_table('pref_jobtitle')
    op.drop_table('pref_jobs')
    op.drop_table('user_resumes')
    op.drop_table('user_onboarding')
    op.drop_table('lookup_jobtype')
    op.drop_table('lookup_jobtitles')
    # ### end Alembic commands ###
